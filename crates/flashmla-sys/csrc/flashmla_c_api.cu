#include "flashmla_c_api.h"

#include <cuda_runtime_api.h>
#include <cutlass/bfloat16.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <string>

#include "params.h"
#include "sm90/decode/dense/splitkv_mla.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm90/prefill/sparse/fwd.h"
#include "smxx/decode/combine/combine.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

namespace {

constexpr float kLog2E = 1.4426950408889634074f;
constexpr int kDecodeSchedMetaI32Len = sizeof(DecodingSchedMeta) / sizeof(int);
constexpr int kDecodeMaxNumSmPartsForCombine = 160;

thread_local std::string g_last_error;

flashmla_status_t set_error(flashmla_status_t status, const std::string& message) {
  g_last_error = message;
  return status;
}

flashmla_status_t set_cuda_error(const char* context, cudaError_t error) {
  std::string message(context);
  message += ": ";
  message += cudaGetErrorString(error);
  return set_error(FLASHMLA_STATUS_CUDA_ERROR, message);
}

flashmla_status_t clear_error() {
  g_last_error.clear();
  return FLASHMLA_STATUS_SUCCESS;
}

bool is_null_required_prefill_pointer(const flashmla_sparse_prefill_params_t* params) {
  return params->q == nullptr ||
         params->kv == nullptr ||
         params->indices == nullptr ||
         params->out == nullptr ||
         params->max_logits == nullptr ||
         params->lse == nullptr;
}

bool checked_mul_size(size_t left, size_t right, size_t* out) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  *out = left * right;
  return true;
}

bool checked_add_size(size_t left, size_t right, size_t* out) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return false;
  }
  *out = left + right;
  return true;
}

int ceil_div_positive(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

bool has_extra_decode_cache(const flashmla_sparse_decode_params_t* params) {
  return params->extra_num_blocks > 0 ||
         params->extra_page_block_size > 0 ||
         params->extra_topk > 0 ||
         params->extra_kv != nullptr ||
         params->extra_indices != nullptr ||
         params->extra_topk_length != nullptr;
}

bool has_extra_decode_plan(const flashmla_sparse_decode_plan_params_t* params) {
  return params->extra_topk > 0 || params->extra_topk_length != nullptr;
}

int sparse_decode_bytes_per_token(int d_qk, int d_v) {
  if (d_qk == 576 && d_v == 512) {
    return 656;
  }
  if (d_qk == 512 && d_v == 512) {
    return 584;
  }
  return -1;
}

flashmla_status_t validate_current_device_sm90(const char* context) {
  int current_device = 0;
  cudaError_t error = cudaGetDevice(&current_device);
  if (error != cudaSuccess) {
    return set_cuda_error("cudaGetDevice failed", error);
  }

  cudaDeviceProp props;
  error = cudaGetDeviceProperties(&props, current_device);
  if (error != cudaSuccess) {
    return set_cuda_error("cudaGetDeviceProperties failed", error);
  }
  if (props.major != 9 || props.minor != 0) {
    return set_error(FLASHMLA_STATUS_UNSUPPORTED_ARCH, context);
  }

  return FLASHMLA_STATUS_SUCCESS;
}

flashmla_status_t validate_sparse_prefill_params(
  const flashmla_sparse_prefill_params_t* params
) {
  if (params == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  if (is_null_required_prefill_pointer(params)) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "q, kv, indices, out, max_logits, and lse must not be null"
    );
  }
  if (params->s_q <= 0 || params->s_kv <= 0 || params->topk <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "s_q, s_kv, and topk must be positive"
    );
  }
  if (params->h_q != 64 && params->h_q != 128) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "h_q must be padded to 64 or 128 for SM90 sparse prefill"
    );
  }
  if (params->h_kv != 1) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "h_kv must be 1 for sparse MLA prefill"
    );
  }
  if (params->d_qk != 512 && params->d_qk != 576) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "d_qk must be 512 or 576"
    );
  }
  if (params->d_v != 512) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "d_v must be 512");
  }
  if (params->topk % 128 != 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "SM90 sparse prefill requires topk to be a positive multiple of 128"
    );
  }
  if (params->num_sm <= 0) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "num_sm must be positive");
  }
  if (params->stride_q_s_q <= 0 || params->stride_q_h_q <= 0 ||
      params->stride_kv_s_kv <= 0 || params->stride_kv_h_kv <= 0 ||
      params->stride_indices_s_q <= 0 || params->stride_indices_h_kv <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "q, kv, and indices strides must be positive"
    );
  }

  return validate_current_device_sm90("sparse prefill wrapper only supports SM90");
}

flashmla_status_t validate_sparse_decode_shape(
  int batch,
  int s_q,
  int h_q,
  int h_kv,
  int d_qk,
  int d_v,
  int topk,
  bool has_extra,
  const void* topk_length,
  const void* extra_topk_length
) {
  if (batch <= 0 || s_q <= 0 || topk <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "batch, s_q, and topk must be positive"
    );
  }
  if (h_q != 64 && h_q != 128) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "h_q must be padded to 64 or 128 for SM90 sparse decode"
    );
  }
  if (h_kv != 1) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "h_kv must be 1 for sparse MLA decode"
    );
  }
  if (d_qk != 512 && d_qk != 576) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "d_qk must be 512 or 576"
    );
  }
  if (d_v != 512) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "d_v must be 512");
  }
  if (topk % 64 != 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "SM90 sparse decode requires topk to be a multiple of 64"
    );
  }
  if (d_qk == 576) {
    if (topk_length != nullptr) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "V32 sparse decode does not support topk_length"
      );
    }
    if (has_extra) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "V32 sparse decode does not support extra KV cache"
      );
    }
  }
  if (!has_extra && extra_topk_length != nullptr) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "extra_topk_length requires an extra KV cache"
    );
  }
  return FLASHMLA_STATUS_SUCCESS;
}

flashmla_status_t fill_sparse_decode_plan_result(
  int batch,
  int s_q,
  int h_q,
  int d_v,
  int num_sm,
  flashmla_sparse_decode_plan_result_t* result
) {
  if (num_sm <= 0) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "num_sm must be positive");
  }

  const int head_factor = h_q / 64;
  const int denom = s_q * head_factor;
  const int num_sm_parts = std::max(num_sm / denom, 1);
  if (num_sm_parts > kDecodeMaxNumSmPartsForCombine) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "computed num_sm_parts exceeds the combine kernel maximum"
    );
  }

  // With one partition every request is unsplit and the decode kernel writes
  // directly to out/lse. Keep one element because the raw launch API requires
  // non-null accumulator pointers, but no accumulator element is accessed.
  size_t lse_accum = 1;
  size_t o_accum = 1;
  if (num_sm_parts > 1) {
    size_t total_splits = static_cast<size_t>(batch) + static_cast<size_t>(num_sm_parts);
    if (!checked_mul_size(total_splits, static_cast<size_t>(s_q), &lse_accum) ||
        !checked_mul_size(lse_accum, static_cast<size_t>(h_q), &lse_accum) ||
        !checked_mul_size(lse_accum, static_cast<size_t>(d_v), &o_accum)) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "sparse decode workspace element count overflow"
      );
    }
  }

  result->num_sm_parts = num_sm_parts;
  result->fixed_overhead_num_blocks = 5;
  result->block_size_topk = 64;
  result->scheduler_metadata_i32_len =
    static_cast<size_t>(num_sm_parts) * static_cast<size_t>(kDecodeSchedMetaI32Len);
  result->num_splits_len = static_cast<size_t>(batch) + 1;
  result->lse_accum_elem_count = lse_accum;
  result->o_accum_elem_count = o_accum;
  return FLASHMLA_STATUS_SUCCESS;
}

flashmla_status_t validate_dense_decode_shape(
  int batch,
  int s_q,
  int h_q,
  int h_k,
  int d_qk,
  int d_v
) {
  if (batch <= 0 || s_q <= 0 || h_q <= 0 || h_k <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "batch, s_q, h_q, and h_k must be positive"
    );
  }
  if (h_q % h_k != 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "h_k must divide h_q for dense decode"
    );
  }
  if (d_qk != 512 && d_qk != 576) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "d_qk must be 512 or 576"
    );
  }
  if (d_v != 512) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "d_v must be 512");
  }

  const int q_heads_per_hk = h_q / h_k;
  if (q_heads_per_hk <= 0 ||
      s_q > std::numeric_limits<int>::max() / q_heads_per_hk) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "q_seq_per_hk overflows int"
    );
  }

  return FLASHMLA_STATUS_SUCCESS;
}

flashmla_status_t fill_dense_decode_plan_result(
  int batch,
  int s_q,
  int h_q,
  int h_k,
  int d_v,
  int num_sm,
  flashmla_dense_decode_plan_result_t* result
) {
  if (num_sm <= 0) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "num_sm must be positive");
  }

  const int q_heads_per_hk = h_q / h_k;
  const int q_seq_per_hk = s_q * q_heads_per_hk;
  const int m_blocks = ceil_div_positive(q_seq_per_hk, 64);
  const int num_sm_parts = std::max(num_sm / h_k / m_blocks, 1);
  if (num_sm_parts > kDecodeMaxNumSmPartsForCombine) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "computed num_sm_parts exceeds the combine kernel maximum"
    );
  }

  size_t total_splits = 0;
  size_t lse_accum = 0;
  size_t o_accum = 0;
  if (!checked_add_size(static_cast<size_t>(batch), static_cast<size_t>(num_sm_parts), &total_splits) ||
      !checked_mul_size(total_splits, static_cast<size_t>(h_k), &lse_accum) ||
      !checked_mul_size(lse_accum, static_cast<size_t>(q_seq_per_hk), &lse_accum) ||
      !checked_mul_size(lse_accum, static_cast<size_t>(d_v), &o_accum)) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "dense decode workspace element count overflow"
    );
  }

  result->num_sm_parts = num_sm_parts;
  result->fixed_overhead_num_blocks = 5;
  result->block_size_n = 64;
  result->q_seq_per_hk = q_seq_per_hk;
  result->scheduler_metadata_i32_len =
    static_cast<size_t>(num_sm_parts) * static_cast<size_t>(kDecodeSchedMetaI32Len);
  result->num_splits_len = static_cast<size_t>(batch) + 1;
  result->lse_accum_elem_count = lse_accum;
  result->o_accum_elem_count = o_accum;
  return FLASHMLA_STATUS_SUCCESS;
}

flashmla_status_t validate_dense_decode_plan_params(
  const flashmla_dense_decode_plan_params_t* params,
  flashmla_dense_decode_plan_result_t* result
) {
  if (params == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  if (result == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "result must not be null");
  }
  if ((params->tile_scheduler_metadata == nullptr) != (params->num_splits == nullptr)) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "tile_scheduler_metadata and num_splits must either both be null or both be non-null"
    );
  }
  if (params->tile_scheduler_metadata != nullptr && params->seqlens_k == nullptr) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "seqlens_k must not be null when generating dense decode scheduler metadata"
    );
  }

  flashmla_status_t status = validate_dense_decode_shape(
    params->batch,
    params->s_q,
    params->h_q,
    params->h_k,
    params->d_qk,
    params->d_v
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  status = fill_dense_decode_plan_result(
    params->batch,
    params->s_q,
    params->h_q,
    params->h_k,
    params->d_v,
    params->num_sm,
    result
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  return validate_current_device_sm90("dense decode wrapper only supports SM90");
}

flashmla_status_t validate_dense_decode_params(
  const flashmla_dense_decode_params_t* params
) {
  if (params == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  if (params->q == nullptr || params->kcache == nullptr || params->seqlens_k == nullptr ||
      params->block_table == nullptr || params->out == nullptr || params->lse == nullptr ||
      params->lse_accum == nullptr || params->o_accum == nullptr ||
      params->tile_scheduler_metadata == nullptr || params->num_splits == nullptr) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "q, kcache, seqlens_k, block_table, out, lse, lse_accum, o_accum, tile_scheduler_metadata, and num_splits must not be null"
    );
  }

  flashmla_status_t status = validate_dense_decode_shape(
    params->batch,
    params->s_q,
    params->h_q,
    params->h_k,
    params->d_qk,
    params->d_v
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  if (params->num_blocks <= 0 || params->page_block_size <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "num_blocks and page_block_size must be positive"
    );
  }
  if (params->page_block_size != 64) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "dense decode requires page_block_size to be 64"
    );
  }
  if (params->num_sm_parts <= 0 || params->num_sm_parts > kDecodeMaxNumSmPartsForCombine) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "num_sm_parts must be between 1 and 160"
    );
  }
  if (params->stride_q_b <= 0 || params->stride_q_row <= 0 ||
      params->stride_q_head <= 0 || params->stride_k_block <= 0 ||
      params->stride_k_row <= 0 || params->stride_k_head <= 0 ||
      params->stride_block_table_b <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "dense decode strides must be positive"
    );
  }

  return validate_current_device_sm90("dense decode wrapper only supports SM90");
}

flashmla_status_t validate_sparse_decode_plan_params(
  const flashmla_sparse_decode_plan_params_t* params,
  flashmla_sparse_decode_plan_result_t* result
) {
  if (params == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  if (result == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "result must not be null");
  }
  if ((params->tile_scheduler_metadata == nullptr) != (params->num_splits == nullptr)) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "tile_scheduler_metadata and num_splits must either both be null or both be non-null"
    );
  }

  flashmla_status_t status = validate_sparse_decode_shape(
    params->batch,
    params->s_q,
    params->h_q,
    params->h_kv,
    params->d_qk,
    params->d_v,
    params->topk,
    has_extra_decode_plan(params),
    params->topk_length,
    params->extra_topk_length
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  status = fill_sparse_decode_plan_result(
    params->batch,
    params->s_q,
    params->h_q,
    params->d_v,
    params->num_sm,
    result
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  return validate_current_device_sm90("sparse decode wrapper only supports SM90");
}

flashmla_status_t validate_sparse_decode_params(
  const flashmla_sparse_decode_params_t* params
) {
  if (params == nullptr) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "params must not be null");
  }
  if (params->q == nullptr || params->kv == nullptr || params->indices == nullptr ||
      params->out == nullptr || params->lse == nullptr || params->lse_accum == nullptr ||
      params->o_accum == nullptr || params->tile_scheduler_metadata == nullptr ||
      params->num_splits == nullptr) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "q, kv, indices, out, lse, lse_accum, o_accum, tile_scheduler_metadata, and num_splits must not be null"
    );
  }

  const bool has_extra = has_extra_decode_cache(params);
  flashmla_status_t status = validate_sparse_decode_shape(
    params->batch,
    params->s_q,
    params->h_q,
    params->h_kv,
    params->d_qk,
    params->d_v,
    params->topk,
    has_extra,
    params->topk_length,
    params->extra_topk_length
  );
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  if (params->num_blocks <= 0 || params->page_block_size <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "num_blocks and page_block_size must be positive"
    );
  }
  if (params->num_sm_parts <= 0 || params->num_sm_parts > kDecodeMaxNumSmPartsForCombine) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "num_sm_parts must be between 1 and 160"
    );
  }
  if (params->stride_q_b <= 0 || params->stride_q_s_q <= 0 || params->stride_q_h_q <= 0 ||
      params->stride_kv_block <= 0 || params->stride_kv_row <= 0 ||
      params->stride_indices_b <= 0 || params->stride_indices_s_q <= 0 ||
      params->stride_lse_b <= 0 || params->stride_lse_s_q <= 0 ||
      params->stride_o_b <= 0 || params->stride_o_s_q <= 0 || params->stride_o_h_q <= 0 ||
      params->stride_lse_accum_split <= 0 || params->stride_lse_accum_s_q <= 0 ||
      params->stride_o_accum_split <= 0 || params->stride_o_accum_s_q <= 0 ||
      params->stride_o_accum_h_q <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "sparse decode strides must be positive"
    );
  }

  const int bytes_per_token = sparse_decode_bytes_per_token(params->d_qk, params->d_v);
  if (bytes_per_token <= 0) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "unsupported sparse decode KV cache layout"
    );
  }
  if (params->stride_kv_row != bytes_per_token) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "kv page rows must be contiguous and match the FlashMLA bytes-per-token layout"
    );
  }

  if (has_extra) {
    if (params->extra_num_blocks <= 0 || params->extra_page_block_size <= 0 ||
        params->extra_topk <= 0 || params->extra_kv == nullptr ||
        params->extra_indices == nullptr) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "extra KV cache requires positive extra dimensions plus extra_kv and extra_indices"
      );
    }
    if (params->stride_extra_kv_block <= 0 || params->stride_extra_kv_row <= 0 ||
        params->stride_extra_indices_b <= 0 || params->stride_extra_indices_s_q <= 0) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "extra sparse decode strides must be positive"
      );
    }
    if (params->stride_extra_kv_row != bytes_per_token) {
      return set_error(
        FLASHMLA_STATUS_INVALID_ARGUMENT,
        "extra kv page rows must be contiguous and match the FlashMLA bytes-per-token layout"
      );
    }
  }

  return validate_current_device_sm90("sparse decode wrapper only supports SM90");
}

}  // namespace

const char* flashmla_last_error(void) {
  return g_last_error.c_str();
}

flashmla_status_t flashmla_get_device_info(
  int device_id,
  int* major,
  int* minor,
  int* num_sms
) {
  if (device_id < 0) {
    return set_error(FLASHMLA_STATUS_INVALID_ARGUMENT, "device_id must be non-negative");
  }
  if (major == nullptr || minor == nullptr || num_sms == nullptr) {
    return set_error(
      FLASHMLA_STATUS_INVALID_ARGUMENT,
      "major, minor, and num_sms must not be null"
    );
  }

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, device_id);
  if (error != cudaSuccess) {
    return set_cuda_error("cudaGetDeviceProperties failed", error);
  }

  *major = props.major;
  *minor = props.minor;
  *num_sms = props.multiProcessorCount;
  return clear_error();
}

flashmla_status_t flashmla_sparse_prefill_bf16(
  const flashmla_sparse_prefill_params_t* params
) {
  flashmla_status_t status = validate_sparse_prefill_params(params);
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  try {
    SparseAttnFwdParams upstream_params = {
      params->s_q,
      params->s_kv,
      params->h_q,
      params->h_kv,
      params->d_qk,
      params->d_v,
      params->topk,
      params->sm_scale,
      params->sm_scale * kLog2E,

      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(params->q)),
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(params->kv)),
      const_cast<int*>(params->indices),
      const_cast<float*>(params->attn_sink),
      const_cast<int*>(params->topk_length),

      params->stride_q_s_q,
      params->stride_q_h_q,
      params->stride_kv_s_kv,
      params->stride_kv_h_kv,
      params->stride_indices_s_q,
      params->stride_indices_h_kv,

      reinterpret_cast<cutlass::bfloat16_t*>(params->out),
      params->max_logits,
      params->lse,

      params->num_sm,
      reinterpret_cast<cudaStream_t>(params->stream)
    };

    sm90::run_fwd_kernel(upstream_params);
  } catch (const std::exception& error) {
    return set_error(FLASHMLA_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(
      FLASHMLA_STATUS_INTERNAL_ERROR,
      "unknown exception while launching sparse prefill"
    );
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return set_cuda_error("sparse prefill launch failed", error);
  }

  return clear_error();
}

flashmla_status_t flashmla_sparse_decode_plan(
  const flashmla_sparse_decode_plan_params_t* params,
  flashmla_sparse_decode_plan_result_t* result
) {
  flashmla_status_t status = validate_sparse_decode_plan_params(params, result);
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  if (params->tile_scheduler_metadata == nullptr) {
    return clear_error();
  }

  try {
    GetDecodeSchedMetaParams upstream_params = {
      params->batch,
      params->s_q,
      result->block_size_topk,
      result->fixed_overhead_num_blocks,
      params->topk,
      params->extra_topk,
      const_cast<int*>(params->topk_length),
      const_cast<int*>(params->extra_topk_length),
      nullptr,
      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata),
      params->num_splits,
      result->num_sm_parts,
      reinterpret_cast<cudaStream_t>(params->stream)
    };
    smxx::decode::run_get_decoding_sched_meta_kernel(upstream_params);
  } catch (const std::exception& error) {
    return set_error(FLASHMLA_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(
      FLASHMLA_STATUS_INTERNAL_ERROR,
      "unknown exception while generating sparse decode metadata"
    );
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return set_cuda_error("sparse decode metadata generation failed", error);
  }

  return clear_error();
}

flashmla_status_t flashmla_dense_decode_plan(
  const flashmla_dense_decode_plan_params_t* params,
  flashmla_dense_decode_plan_result_t* result
) {
  flashmla_status_t status = validate_dense_decode_plan_params(params, result);
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  if (params->tile_scheduler_metadata == nullptr) {
    return clear_error();
  }

  try {
    GetDecodeSchedMetaParams upstream_params = {
      params->batch,
      params->s_q,
      result->block_size_n,
      result->fixed_overhead_num_blocks,
      -1,
      -1,
      nullptr,
      nullptr,
      const_cast<int*>(params->seqlens_k),
      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata),
      params->num_splits,
      result->num_sm_parts,
      reinterpret_cast<cudaStream_t>(params->stream)
    };
    smxx::decode::run_get_decoding_sched_meta_kernel(upstream_params);
  } catch (const std::exception& error) {
    return set_error(FLASHMLA_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(
      FLASHMLA_STATUS_INTERNAL_ERROR,
      "unknown exception while generating dense decode metadata"
    );
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return set_cuda_error("dense decode metadata generation failed", error);
  }

  return clear_error();
}

flashmla_status_t flashmla_sparse_decode_bf16_fp8(
  const flashmla_sparse_decode_params_t* params
) {
  flashmla_status_t status = validate_sparse_decode_params(params);
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  const ModelType model_type = params->d_qk == 576 ? ModelType::V32 : ModelType::MODEL1;

  try {
    SparseAttnDecodeParams upstream_params = {
      params->batch,
      params->s_q,
      params->h_q,
      params->h_kv,
      params->d_qk,
      params->d_v,
      params->sm_scale,
      params->sm_scale * kLog2E,
      params->num_blocks,
      params->page_block_size,
      params->topk,
      model_type,

      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(params->q)),
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(params->kv)),
      const_cast<int*>(params->indices),
      const_cast<int*>(params->topk_length),
      const_cast<float*>(params->attn_sink),
      params->lse,
      reinterpret_cast<cutlass::bfloat16_t*>(params->out),

      params->extra_num_blocks,
      params->extra_page_block_size,
      params->extra_topk,
      reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(params->extra_kv)),
      const_cast<int*>(params->extra_indices),
      const_cast<int*>(params->extra_topk_length),

      params->stride_q_b,
      params->stride_q_s_q,
      params->stride_q_h_q,
      params->stride_kv_block,
      params->stride_kv_row,
      params->stride_indices_b,
      params->stride_indices_s_q,
      params->stride_lse_b,
      params->stride_lse_s_q,
      params->stride_o_b,
      params->stride_o_s_q,
      params->stride_o_h_q,
      params->stride_extra_kv_block,
      params->stride_extra_kv_row,
      params->stride_extra_indices_b,
      params->stride_extra_indices_s_q,
      reinterpret_cast<cudaStream_t>(params->stream),

      params->lse_accum,
      params->o_accum,
      params->stride_lse_accum_split,
      params->stride_lse_accum_s_q,
      params->stride_o_accum_split,
      params->stride_o_accum_s_q,
      params->stride_o_accum_h_q,
      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata),
      params->num_splits,
      params->num_sm_parts
    };

    if (model_type == ModelType::V32) {
      if (params->h_q == 64) {
        sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(
          upstream_params
        );
      } else {
        sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 128>(
          upstream_params
        );
      }
    } else {
      if (params->h_q == 64) {
        sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(
          upstream_params
        );
      } else {
        sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 128>(
          upstream_params
        );
      }
    }

    // Preserve upstream's PDL combine launch even for one-part schedules. Its
    // CTAs return without touching the scalar accumulators, while retaining
    // upstream launch and ordering semantics for downstream work.
    CombineParams combine_params = {
      params->batch,
      params->s_q,
      params->h_q,
      params->d_v,

      params->lse,
      params->out,
      params->stride_lse_b,
      params->stride_lse_s_q,
      params->stride_o_b,
      params->stride_o_s_q,
      params->stride_o_h_q,

      params->lse_accum,
      params->o_accum,
      params->stride_lse_accum_split,
      params->stride_lse_accum_s_q,
      params->stride_o_accum_split,
      params->stride_o_accum_s_q,
      params->stride_o_accum_h_q,

      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata),
      params->num_splits,
      params->num_sm_parts,

      const_cast<float*>(params->attn_sink),
      reinterpret_cast<cudaStream_t>(params->stream)
    };
    smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);
  } catch (const std::exception& error) {
    return set_error(FLASHMLA_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(
      FLASHMLA_STATUS_INTERNAL_ERROR,
      "unknown exception while launching sparse decode"
    );
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return set_cuda_error("sparse decode launch failed", error);
  }

  return clear_error();
}

flashmla_status_t flashmla_dense_decode_bf16(
  const flashmla_dense_decode_params_t* params
) {
  flashmla_status_t status = validate_dense_decode_params(params);
  if (status != FLASHMLA_STATUS_SUCCESS) {
    return status;
  }

  const int q_heads_per_hk = params->h_q / params->h_k;
  const int q_seq_per_hk = params->s_q * q_heads_per_hk;
  const int total_num_splits = params->batch + params->num_sm_parts;

  try {
    DenseAttnDecodeParams upstream_params;
    upstream_params.b = params->batch;
    upstream_params.s_q = params->s_q;
    upstream_params.q_seq_per_hk = q_seq_per_hk;
    upstream_params.d = params->d_qk;
    upstream_params.d_v = params->d_v;
    upstream_params.h_q = params->h_q;
    upstream_params.h_k = params->h_k;
    upstream_params.num_blocks = params->num_blocks;
    upstream_params.q_head_per_hk = q_heads_per_hk;
    upstream_params.is_causal = params->s_q == 1 ? false : params->is_causal != 0;
    upstream_params.scale_softmax = params->sm_scale;
    upstream_params.scale_softmax_log2 = params->sm_scale * kLog2E;

    upstream_params.q_ptr = const_cast<void*>(params->q);
    upstream_params.k_ptr = const_cast<void*>(params->kcache);
    upstream_params.o_ptr = params->out;
    upstream_params.softmax_lse_ptr = params->lse;

    upstream_params.q_batch_stride = params->stride_q_b;
    upstream_params.k_batch_stride = params->stride_k_block;
    upstream_params.o_batch_stride = params->h_k * q_seq_per_hk * params->d_v;
    upstream_params.q_row_stride = params->stride_q_row;
    upstream_params.k_row_stride = params->stride_k_row;
    upstream_params.o_row_stride = params->d_v;
    upstream_params.q_head_stride = params->stride_q_head;
    upstream_params.k_head_stride = params->stride_k_head;
    upstream_params.o_head_stride = q_seq_per_hk * params->d_v;

    upstream_params.block_table = const_cast<int*>(params->block_table);
    upstream_params.block_table_batch_stride = params->stride_block_table_b;
    upstream_params.page_block_size = params->page_block_size;
    upstream_params.seqlens_k_ptr = const_cast<int*>(params->seqlens_k);

    upstream_params.tile_scheduler_metadata_ptr =
      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata);
    upstream_params.num_sm_parts = params->num_sm_parts;
    upstream_params.num_splits_ptr = params->num_splits;

    upstream_params.total_num_splits = total_num_splits;
    upstream_params.softmax_lseaccum_ptr = params->lse_accum;
    upstream_params.oaccum_ptr = params->o_accum;

    upstream_params.stream = reinterpret_cast<cudaStream_t>(params->stream);

    sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(upstream_params);

    CombineParams combine_params = {
      params->batch,
      params->s_q,
      params->h_q,
      params->d_v,

      params->lse,
      params->out,
      params->h_k * q_seq_per_hk,
      params->h_q,
      params->h_q * params->s_q * params->d_v,
      params->h_q * params->d_v,
      params->d_v,

      params->lse_accum,
      params->o_accum,
      params->h_k * q_seq_per_hk,
      params->h_q,
      params->h_q * params->s_q * params->d_v,
      params->h_q * params->d_v,
      params->d_v,

      reinterpret_cast<DecodingSchedMeta*>(params->tile_scheduler_metadata),
      params->num_splits,
      params->num_sm_parts,

      nullptr,
      reinterpret_cast<cudaStream_t>(params->stream)
    };
    smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);
  } catch (const std::exception& error) {
    return set_error(FLASHMLA_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return set_error(
      FLASHMLA_STATUS_INTERNAL_ERROR,
      "unknown exception while launching dense decode"
    );
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return set_cuda_error("dense decode launch failed", error);
  }

  return clear_error();
}

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum flashmla_status_t {
  FLASHMLA_STATUS_SUCCESS = 0,
  FLASHMLA_STATUS_INVALID_ARGUMENT = 1,
  FLASHMLA_STATUS_UNSUPPORTED_ARCH = 2,
  FLASHMLA_STATUS_CUDA_ERROR = 3,
  FLASHMLA_STATUS_INTERNAL_ERROR = 4,
} flashmla_status_t;

typedef void* flashmla_cuda_stream_t;

typedef struct flashmla_sparse_prefill_params_t {
  int s_q;
  int s_kv;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int topk;
  float sm_scale;

  const void* q;
  const void* kv;
  const int* indices;
  const float* attn_sink;
  const int* topk_length;

  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  void* out;
  float* max_logits;
  float* lse;

  int num_sm;
  flashmla_cuda_stream_t stream;
} flashmla_sparse_prefill_params_t;

typedef struct flashmla_sparse_decode_plan_params_t {
  int batch;
  int s_q;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int topk;
  int extra_topk;

  const int* topk_length;
  const int* extra_topk_length;

  int* tile_scheduler_metadata;
  int* num_splits;

  int num_sm;
  flashmla_cuda_stream_t stream;
} flashmla_sparse_decode_plan_params_t;

typedef struct flashmla_sparse_decode_plan_result_t {
  int num_sm_parts;
  int fixed_overhead_num_blocks;
  int block_size_topk;
  size_t scheduler_metadata_i32_len;
  size_t num_splits_len;
  size_t lse_accum_elem_count;
  size_t o_accum_elem_count;
} flashmla_sparse_decode_plan_result_t;

typedef struct flashmla_dense_decode_plan_params_t {
  int batch;
  int s_q;
  int h_q;
  int h_k;
  int d_qk;
  int d_v;

  const int* seqlens_k;

  int* tile_scheduler_metadata;
  int* num_splits;

  int num_sm;
  flashmla_cuda_stream_t stream;
} flashmla_dense_decode_plan_params_t;

typedef struct flashmla_dense_decode_plan_result_t {
  int num_sm_parts;
  int fixed_overhead_num_blocks;
  int block_size_n;
  int q_seq_per_hk;
  size_t scheduler_metadata_i32_len;
  size_t num_splits_len;
  size_t lse_accum_elem_count;
  size_t o_accum_elem_count;
} flashmla_dense_decode_plan_result_t;

typedef struct flashmla_dense_decode_params_t {
  int batch;
  int s_q;
  int h_q;
  int h_k;
  int d_qk;
  int d_v;
  int num_blocks;
  int page_block_size;
  int is_causal;
  float sm_scale;

  const void* q;
  const void* kcache;
  const int* seqlens_k;
  const int* block_table;

  void* out;
  float* lse;
  float* lse_accum;
  float* o_accum;

  int stride_q_b;
  int stride_q_row;
  int stride_q_head;
  int stride_k_block;
  int stride_k_row;
  int stride_k_head;
  int stride_block_table_b;

  int* tile_scheduler_metadata;
  int* num_splits;
  int num_sm_parts;

  flashmla_cuda_stream_t stream;
} flashmla_dense_decode_params_t;

typedef struct flashmla_sparse_decode_params_t {
  int batch;
  int s_q;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int num_blocks;
  int page_block_size;
  int topk;
  float sm_scale;

  const void* q;
  const void* kv;
  const int* indices;
  const int* topk_length;
  const float* attn_sink;
  void* out;
  float* lse;

  int extra_num_blocks;
  int extra_page_block_size;
  int extra_topk;
  const void* extra_kv;
  const int* extra_indices;
  const int* extra_topk_length;

  int stride_q_b;
  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_block;
  int stride_kv_row;
  int stride_indices_b;
  int stride_indices_s_q;
  int stride_lse_b;
  int stride_lse_s_q;
  int stride_o_b;
  int stride_o_s_q;
  int stride_o_h_q;
  int stride_extra_kv_block;
  int stride_extra_kv_row;
  int stride_extra_indices_b;
  int stride_extra_indices_s_q;

  float* lse_accum;
  float* o_accum;
  int stride_lse_accum_split;
  int stride_lse_accum_s_q;
  int stride_o_accum_split;
  int stride_o_accum_s_q;
  int stride_o_accum_h_q;

  int* tile_scheduler_metadata;
  int* num_splits;
  int num_sm_parts;

  flashmla_cuda_stream_t stream;
} flashmla_sparse_decode_params_t;

const char* flashmla_last_error(void);

flashmla_status_t flashmla_get_device_info(
  int device_id,
  int* major,
  int* minor,
  int* num_sms
);

flashmla_status_t flashmla_sparse_prefill_bf16(
  const flashmla_sparse_prefill_params_t* params
);

flashmla_status_t flashmla_sparse_decode_plan(
  const flashmla_sparse_decode_plan_params_t* params,
  flashmla_sparse_decode_plan_result_t* result
);

flashmla_status_t flashmla_sparse_decode_bf16_fp8(
  const flashmla_sparse_decode_params_t* params
);

flashmla_status_t flashmla_dense_decode_plan(
  const flashmla_dense_decode_plan_params_t* params,
  flashmla_dense_decode_plan_result_t* result
);

flashmla_status_t flashmla_dense_decode_bf16(
  const flashmla_dense_decode_params_t* params
);

#ifdef __cplusplus
}
#endif

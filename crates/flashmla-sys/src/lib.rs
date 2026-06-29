#![allow(non_camel_case_types)]
//! Raw FlashMLA C ABI bindings.

use core::ffi::{c_char, c_int, c_void};

/// Opaque CUDA stream handle used by the FlashMLA C ABI.
pub type cudaStream_t = *mut c_void;

/// FlashMLA source root selected by the build script.
pub const FLASHMLA_SOURCE_ROOT: &str = env!("FLASHMLA_SOURCE_ROOT");

/// Status code returned by FlashMLA C ABI functions.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum flashmla_status_t {
    /// The operation completed successfully.
    FLASHMLA_STATUS_SUCCESS = 0,
    /// A caller-provided argument was invalid.
    FLASHMLA_STATUS_INVALID_ARGUMENT = 1,
    /// The selected GPU architecture is not supported.
    FLASHMLA_STATUS_UNSUPPORTED_ARCH = 2,
    /// CUDA returned an error.
    FLASHMLA_STATUS_CUDA_ERROR = 3,
    /// FlashMLA hit an unexpected internal error.
    FLASHMLA_STATUS_INTERNAL_ERROR = 4,
}

/// Raw BF16 sparse prefill launch parameters.
///
/// All pointers are CUDA device pointers unless otherwise noted. Strides are element strides, not
/// byte strides. Optional pointers may be null.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct flashmla_sparse_prefill_params_t {
    /// Query sequence length.
    pub s_q: c_int,
    /// KV sequence length.
    pub s_kv: c_int,
    /// Query head count after padding.
    pub h_q: c_int,
    /// KV head count.
    pub h_kv: c_int,
    /// Query/key head dimension.
    pub d_qk: c_int,
    /// Value head dimension.
    pub d_v: c_int,
    /// Number of sparse KV indices per query.
    pub topk: c_int,
    /// Softmax scale applied to QK logits.
    pub sm_scale: f32,

    /// BF16 query pointer shaped `[s_q, h_q, d_qk]`.
    pub q: *const c_void,
    /// BF16 KV pointer shaped `[s_kv, h_kv, d_qk]`.
    pub kv: *const c_void,
    /// I32 sparse index pointer shaped `[s_q, h_kv, topk]`.
    pub indices: *const c_int,
    /// Optional F32 attention sink pointer shaped `[h_q]`.
    pub attn_sink: *const f32,
    /// Optional I32 top-k length pointer shaped `[s_q]`.
    pub topk_length: *const c_int,

    /// Element stride between query sequence positions.
    pub stride_q_s_q: c_int,
    /// Element stride between query heads.
    pub stride_q_h_q: c_int,
    /// Element stride between KV sequence positions.
    pub stride_kv_s_kv: c_int,
    /// Element stride between KV heads.
    pub stride_kv_h_kv: c_int,
    /// Element stride between sparse-index query positions.
    pub stride_indices_s_q: c_int,
    /// Element stride between sparse-index KV heads.
    pub stride_indices_h_kv: c_int,

    /// BF16 output pointer shaped `[s_q, h_q, d_v]`.
    pub out: *mut c_void,
    /// F32 max-logits output pointer shaped `[s_q, h_q]`.
    pub max_logits: *mut f32,
    /// F32 log-sum-exp output pointer shaped `[s_q, h_q]`.
    pub lse: *mut f32,

    /// Number of SMs to pass to the upstream kernel.
    pub num_sm: c_int,
    /// CUDA stream used for the kernel launch.
    pub stream: cudaStream_t,
}

/// Raw sparse decode scheduler metadata generation parameters.
///
/// All pointers are CUDA device pointers unless otherwise noted. `tile_scheduler_metadata` and
/// `num_splits` may both be null to request size calculation without launching the metadata
/// kernel. If either output pointer is non-null, both must be valid writable buffers sized from
/// `flashmla_sparse_decode_plan_result_t`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct flashmla_sparse_decode_plan_params_t {
    /// Batch size.
    pub batch: c_int,
    /// Query sequence length.
    pub s_q: c_int,
    /// Query head count after padding.
    pub h_q: c_int,
    /// KV head count.
    pub h_kv: c_int,
    /// Query/key head dimension.
    pub d_qk: c_int,
    /// Value head dimension.
    pub d_v: c_int,
    /// Number of sparse KV indices per query.
    pub topk: c_int,
    /// Number of extra sparse KV indices per query, or zero when no extra cache is used.
    pub extra_topk: c_int,

    /// Optional I32 top-k length pointer shaped `[batch]`.
    pub topk_length: *const c_int,
    /// Optional I32 extra top-k length pointer shaped `[batch]`.
    pub extra_topk_length: *const c_int,

    /// Optional writable I32 scheduler metadata buffer.
    pub tile_scheduler_metadata: *mut c_int,
    /// Optional writable I32 split-offset buffer shaped `[batch + 1]`.
    pub num_splits: *mut c_int,

    /// Number of SMs on the target CUDA device.
    pub num_sm: c_int,
    /// CUDA stream used for optional metadata generation.
    pub stream: cudaStream_t,
}

/// Workspace sizing result for sparse decode.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct flashmla_sparse_decode_plan_result_t {
    /// Number of SM partitions used by split-KV decode.
    pub num_sm_parts: c_int,
    /// Fixed scheduler overhead, in top-k blocks.
    pub fixed_overhead_num_blocks: c_int,
    /// Top-k block size used by the scheduler.
    pub block_size_topk: c_int,
    /// Required I32 elements in `tile_scheduler_metadata`.
    pub scheduler_metadata_i32_len: usize,
    /// Required I32 elements in `num_splits`.
    pub num_splits_len: usize,
    /// Required F32 elements in `lse_accum`.
    pub lse_accum_elem_count: usize,
    /// Required F32 elements in `o_accum`.
    pub o_accum_elem_count: usize,
}

/// Raw dense decode scheduler metadata generation parameters.
///
/// All pointers are CUDA device pointers unless otherwise noted. `seqlens_k` is required when
/// `tile_scheduler_metadata` and `num_splits` are non-null, and is shaped `[batch]` with I32
/// sequence lengths. The metadata outputs may both be null to request size calculation without
/// launching the metadata kernel.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct flashmla_dense_decode_plan_params_t {
    /// Batch size.
    pub batch: c_int,
    /// Query sequence length before query-head packing.
    pub s_q: c_int,
    /// Query head count.
    pub h_q: c_int,
    /// KV head count.
    pub h_k: c_int,
    /// Query/key head dimension.
    pub d_qk: c_int,
    /// Value head dimension.
    pub d_v: c_int,

    /// Optional I32 KV sequence lengths shaped `[batch]`.
    pub seqlens_k: *const c_int,

    /// Optional writable I32 scheduler metadata buffer.
    pub tile_scheduler_metadata: *mut c_int,
    /// Optional writable I32 split-offset buffer shaped `[batch + 1]`.
    pub num_splits: *mut c_int,

    /// Number of SMs on the target CUDA device.
    pub num_sm: c_int,
    /// CUDA stream used for optional metadata generation.
    pub stream: cudaStream_t,
}

/// Workspace sizing result for dense decode.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct flashmla_dense_decode_plan_result_t {
    /// Number of SM partitions used by split-KV decode.
    pub num_sm_parts: c_int,
    /// Fixed scheduler overhead, in KV blocks.
    pub fixed_overhead_num_blocks: c_int,
    /// KV block size used by the scheduler.
    pub block_size_n: c_int,
    /// Packed query sequence length per KV head, equal to `s_q * h_q / h_k`.
    pub q_seq_per_hk: c_int,
    /// Required I32 elements in `tile_scheduler_metadata`.
    pub scheduler_metadata_i32_len: usize,
    /// Required I32 elements in `num_splits`.
    pub num_splits_len: usize,
    /// Required F32 elements in dense internal LSE accumulation workspace.
    pub lse_accum_elem_count: usize,
    /// Required F32 elements in dense internal output accumulation workspace.
    pub o_accum_elem_count: usize,
}

/// Raw BF16 dense decode launch parameters.
///
/// `q` must use FlashMLA's packed query layout `[batch, s_q * h_q / h_k, h_k, d_qk]`. `kcache`
/// is BF16 and shaped `[num_blocks, page_block_size, h_k, d_qk]`. `out`, `lse`, `lse_accum`, and
/// `o_accum` use the upstream dense internal layouts documented by the safe wrapper layer. Strides
/// are element strides. The caller owns all input, output, workspace, metadata, and stream
/// lifetimes. Launches are enqueued on `stream` and are not synchronized by this function.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct flashmla_dense_decode_params_t {
    /// Batch size.
    pub batch: c_int,
    /// Query sequence length before query-head packing.
    pub s_q: c_int,
    /// Query head count.
    pub h_q: c_int,
    /// KV head count.
    pub h_k: c_int,
    /// Query/key head dimension.
    pub d_qk: c_int,
    /// Value head dimension.
    pub d_v: c_int,
    /// Number of KV cache pages.
    pub num_blocks: c_int,
    /// Tokens per KV cache page. FlashMLA dense decode currently requires `64`.
    pub page_block_size: c_int,
    /// Non-zero to apply causal masking. Ignored when `s_q == 1`.
    pub is_causal: c_int,
    /// Softmax scale applied to QK logits.
    pub sm_scale: f32,

    /// BF16 packed query pointer shaped `[batch, q_seq_per_hk, h_k, d_qk]`.
    pub q: *const c_void,
    /// BF16 KV cache pointer shaped `[num_blocks, page_block_size, h_k, d_qk]`.
    pub kcache: *const c_void,
    /// I32 KV sequence lengths shaped `[batch]`.
    pub seqlens_k: *const c_int,
    /// I32 block table pointer shaped `[batch, max_num_blocks_per_seq]`.
    pub block_table: *const c_int,

    /// BF16 internal output buffer shaped `[batch, h_k, q_seq_per_hk, d_v]`.
    pub out: *mut c_void,
    /// F32 internal LSE buffer shaped `[batch, h_k, q_seq_per_hk]`.
    pub lse: *mut f32,
    /// F32 split-KV LSE accumulation buffer shaped `[batch + num_sm_parts, h_k, q_seq_per_hk]`.
    pub lse_accum: *mut f32,
    /// F32 split-KV output accumulation buffer shaped `[batch + num_sm_parts, h_k, q_seq_per_hk, d_v]`.
    pub o_accum: *mut f32,

    /// Element stride between packed query batches.
    pub stride_q_b: c_int,
    /// Element stride between packed query rows.
    pub stride_q_row: c_int,
    /// Element stride between packed query KV heads.
    pub stride_q_head: c_int,
    /// Element stride between KV cache pages.
    pub stride_k_block: c_int,
    /// Element stride between KV cache rows.
    pub stride_k_row: c_int,
    /// Element stride between KV cache heads.
    pub stride_k_head: c_int,
    /// Element stride between block-table batches.
    pub stride_block_table_b: c_int,

    /// I32 scheduler metadata buffer generated by `flashmla_dense_decode_plan`.
    pub tile_scheduler_metadata: *mut c_int,
    /// I32 split-offset buffer generated by `flashmla_dense_decode_plan`.
    pub num_splits: *mut c_int,
    /// Number of SM partitions from the plan result.
    pub num_sm_parts: c_int,

    /// CUDA stream used for dense decode and combine launches.
    pub stream: cudaStream_t,
}

/// Raw BF16-query and FP8-cache sparse decode launch parameters.
///
/// `q`, `out`, and the final output dtype are BF16. `kv` and `extra_kv` point at FlashMLA's
/// packed FP8 KV-cache byte layout. `indices`, `topk_length`, `extra_indices`,
/// `extra_topk_length`, `tile_scheduler_metadata`, and `num_splits` are I32 buffers. `lse`,
/// `lse_accum`, and `o_accum` are F32 buffers. Strides are element strides for each tensor's
/// dtype, which means packed FP8 KV strides are byte strides.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct flashmla_sparse_decode_params_t {
    /// Batch size.
    pub batch: c_int,
    /// Query sequence length.
    pub s_q: c_int,
    /// Query head count after padding.
    pub h_q: c_int,
    /// KV head count.
    pub h_kv: c_int,
    /// Query/key head dimension.
    pub d_qk: c_int,
    /// Value head dimension.
    pub d_v: c_int,
    /// Number of KV cache pages.
    pub num_blocks: c_int,
    /// Tokens per KV cache page.
    pub page_block_size: c_int,
    /// Number of sparse KV indices per query.
    pub topk: c_int,
    /// Softmax scale applied to QK logits.
    pub sm_scale: f32,

    /// BF16 query pointer shaped `[batch, s_q, h_q, d_qk]`.
    pub q: *const c_void,
    /// Packed FP8 KV cache pointer shaped `[num_blocks, page_block_size, h_kv, bytes_per_token]`.
    pub kv: *const c_void,
    /// I32 sparse index pointer shaped `[batch, s_q, topk]`.
    pub indices: *const c_int,
    /// Optional I32 top-k length pointer shaped `[batch]`.
    pub topk_length: *const c_int,
    /// Optional F32 attention sink pointer shaped `[h_q]`.
    pub attn_sink: *const f32,
    /// BF16 output pointer shaped `[batch, s_q, h_q, d_v]`.
    pub out: *mut c_void,
    /// F32 log-sum-exp output pointer shaped `[batch, s_q, h_q]`.
    pub lse: *mut f32,

    /// Number of extra KV cache pages, or zero when no extra cache is used.
    pub extra_num_blocks: c_int,
    /// Tokens per extra KV cache page, or zero when no extra cache is used.
    pub extra_page_block_size: c_int,
    /// Number of extra sparse KV indices per query, or zero when no extra cache is used.
    pub extra_topk: c_int,
    /// Optional packed FP8 extra KV cache pointer.
    pub extra_kv: *const c_void,
    /// Optional I32 extra sparse index pointer shaped `[batch, s_q, extra_topk]`.
    pub extra_indices: *const c_int,
    /// Optional I32 extra top-k length pointer shaped `[batch]`.
    pub extra_topk_length: *const c_int,

    /// Element stride between query batches.
    pub stride_q_b: c_int,
    /// Element stride between query positions.
    pub stride_q_s_q: c_int,
    /// Element stride between query heads.
    pub stride_q_h_q: c_int,
    /// Byte stride between KV cache pages.
    pub stride_kv_block: c_int,
    /// Byte stride between KV cache rows.
    pub stride_kv_row: c_int,
    /// Element stride between sparse-index batches.
    pub stride_indices_b: c_int,
    /// Element stride between sparse-index positions.
    pub stride_indices_s_q: c_int,
    /// Element stride between LSE batches.
    pub stride_lse_b: c_int,
    /// Element stride between LSE positions.
    pub stride_lse_s_q: c_int,
    /// Element stride between output batches.
    pub stride_o_b: c_int,
    /// Element stride between output positions.
    pub stride_o_s_q: c_int,
    /// Element stride between output heads.
    pub stride_o_h_q: c_int,
    /// Byte stride between extra KV cache pages.
    pub stride_extra_kv_block: c_int,
    /// Byte stride between extra KV cache rows.
    pub stride_extra_kv_row: c_int,
    /// Element stride between extra sparse-index batches.
    pub stride_extra_indices_b: c_int,
    /// Element stride between extra sparse-index positions.
    pub stride_extra_indices_s_q: c_int,

    /// F32 split-KV LSE accumulation buffer.
    pub lse_accum: *mut f32,
    /// F32 split-KV output accumulation buffer.
    pub o_accum: *mut f32,
    /// Element stride between LSE accumulation splits.
    pub stride_lse_accum_split: c_int,
    /// Element stride between LSE accumulation positions.
    pub stride_lse_accum_s_q: c_int,
    /// Element stride between output accumulation splits.
    pub stride_o_accum_split: c_int,
    /// Element stride between output accumulation positions.
    pub stride_o_accum_s_q: c_int,
    /// Element stride between output accumulation heads.
    pub stride_o_accum_h_q: c_int,

    /// I32 scheduler metadata buffer generated by `flashmla_sparse_decode_plan`.
    pub tile_scheduler_metadata: *mut c_int,
    /// I32 split-offset buffer generated by `flashmla_sparse_decode_plan`.
    pub num_splits: *mut c_int,
    /// Number of SM partitions from the plan result.
    pub num_sm_parts: c_int,

    /// CUDA stream used for sparse decode and combine launches.
    pub stream: cudaStream_t,
}

unsafe extern "C" {
    /// Returns a thread-local error string for the most recent non-success C ABI call.
    pub fn flashmla_last_error() -> *const c_char;

    /// Queries CUDA device compute capability and SM count.
    pub fn flashmla_get_device_info(
        device_id: c_int,
        major: *mut c_int,
        minor: *mut c_int,
        num_sms: *mut c_int,
    ) -> flashmla_status_t;

    /// Launches the SM90 BF16 sparse prefill kernel.
    ///
    /// The caller owns all input, output, and stream lifetimes. On success the launch is enqueued
    /// on `params.stream`; this function does not synchronize the stream.
    pub fn flashmla_sparse_prefill_bf16(
        params: *const flashmla_sparse_prefill_params_t,
    ) -> flashmla_status_t;

    /// Computes sparse decode workspace sizes and optionally generates scheduler metadata.
    ///
    /// If `params.tile_scheduler_metadata` and `params.num_splits` are both null, only `result`
    /// is populated. Otherwise both output pointers must be writable CUDA device buffers and the
    /// metadata kernel is enqueued on `params.stream`.
    pub fn flashmla_sparse_decode_plan(
        params: *const flashmla_sparse_decode_plan_params_t,
        result: *mut flashmla_sparse_decode_plan_result_t,
    ) -> flashmla_status_t;

    /// Launches SM90 sparse BF16-query / FP8-cache decode followed by BF16 combine.
    ///
    /// The caller owns all input, output, workspace, metadata, and stream lifetimes. On success
    /// the launches are enqueued on `params.stream`; this function does not synchronize.
    pub fn flashmla_sparse_decode_bf16_fp8(
        params: *const flashmla_sparse_decode_params_t,
    ) -> flashmla_status_t;

    /// Computes dense decode workspace sizes and optionally generates scheduler metadata.
    ///
    /// If `params.tile_scheduler_metadata` and `params.num_splits` are both null, only `result`
    /// is populated. Otherwise both output pointers must be writable CUDA device buffers,
    /// `params.seqlens_k` must be a valid CUDA device buffer, and the metadata kernel is enqueued
    /// on `params.stream`.
    pub fn flashmla_dense_decode_plan(
        params: *const flashmla_dense_decode_plan_params_t,
        result: *mut flashmla_dense_decode_plan_result_t,
    ) -> flashmla_status_t;

    /// Launches SM90 BF16 dense decode followed by BF16 combine.
    ///
    /// The caller owns all input, output, workspace, metadata, and stream lifetimes. On success
    /// the launches are enqueued on `params.stream`; this function does not synchronize.
    pub fn flashmla_dense_decode_bf16(
        params: *const flashmla_dense_decode_params_t,
    ) -> flashmla_status_t;
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn links_c_abi_symbols() {
        let message = unsafe { CStr::from_ptr(flashmla_last_error()) };
        assert!(message.to_str().unwrap().is_empty());

        let status = unsafe {
            flashmla_get_device_info(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);

        let status = unsafe { flashmla_sparse_prefill_bf16(std::ptr::null()) };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);

        let status = unsafe { flashmla_sparse_decode_plan(std::ptr::null(), std::ptr::null_mut()) };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);

        let status = unsafe { flashmla_sparse_decode_bf16_fp8(std::ptr::null()) };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);

        let status = unsafe { flashmla_dense_decode_plan(std::ptr::null(), std::ptr::null_mut()) };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);

        let status = unsafe { flashmla_dense_decode_bf16(std::ptr::null()) };
        assert_eq!(status, flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT);
    }
}

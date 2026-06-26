use std::ffi::c_void;

use flashmla_sys::{
    cudaStream_t, flashmla_sparse_decode_bf16_fp8 as sys_sparse_decode_bf16_fp8,
    flashmla_sparse_decode_params_t, flashmla_sparse_decode_plan as sys_sparse_decode_plan,
    flashmla_sparse_decode_plan_params_t, flashmla_sparse_decode_plan_result_t, flashmla_status_t,
};

use crate::{Error, Result};

/// Runtime options for sparse decode.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SparseDecodeConfig {
    /// Softmax scale applied to QK logits.
    pub softmax_scale: f32,
    /// Value head dimension. FlashMLA sparse MLA currently requires `512`.
    pub d_v: usize,
    /// Whether higher-level integrations should pad query heads before launch.
    pub pad_heads: bool,
}

impl Default for SparseDecodeConfig {
    fn default() -> Self {
        Self {
            softmax_scale: 1.0,
            d_v: 512,
            pad_heads: true,
        }
    }
}

/// Shape parameters for sparse decode.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseDecodeDims {
    /// Batch size.
    pub batch: usize,
    /// Query sequence length.
    pub s_q: usize,
    /// Query head count after any required padding.
    pub h_q: usize,
    /// KV head count. Sparse MLA currently expects MQA-style `1`.
    pub h_kv: usize,
    /// Query/key head dimension. FlashMLA supports `512` and `576`.
    pub d_qk: usize,
    /// Value head dimension. FlashMLA sparse MLA currently requires `512`.
    pub d_v: usize,
    /// Number of pages in the packed KV cache.
    pub num_blocks: usize,
    /// Number of tokens in each packed KV cache page.
    pub page_block_size: usize,
    /// Number of sparse KV indices per query.
    pub topk: usize,
    /// Number of pages in the optional extra KV cache.
    pub extra_num_blocks: usize,
    /// Number of tokens in each optional extra KV cache page.
    pub extra_page_block_size: usize,
    /// Number of sparse extra KV indices per query.
    pub extra_topk: usize,
}

impl SparseDecodeDims {
    /// Validates architecture-independent sparse decode shape constraints.
    pub fn validate(self) -> Result<()> {
        if self.batch == 0 || self.s_q == 0 || self.topk == 0 {
            return Err(Error::InvalidArgument(
                "batch, s_q, and topk must be non-zero".to_string(),
            ));
        }
        if self.num_blocks == 0 || self.page_block_size == 0 {
            return Err(Error::InvalidArgument(
                "num_blocks and page_block_size must be non-zero".to_string(),
            ));
        }
        if self.d_qk != 512 && self.d_qk != 576 {
            return Err(Error::InvalidArgument(format!(
                "d_qk must be 512 or 576, got {}",
                self.d_qk
            )));
        }
        if self.d_v != 512 {
            return Err(Error::InvalidArgument(format!(
                "d_v must be 512, got {}",
                self.d_v
            )));
        }
        if self.h_q != 64 && self.h_q != 128 {
            return Err(Error::InvalidArgument(format!(
                "h_q must be padded to 64 or 128, got {}",
                self.h_q
            )));
        }
        if self.h_kv != 1 {
            return Err(Error::InvalidArgument(format!(
                "h_kv must be 1 for sparse MLA, got {}",
                self.h_kv
            )));
        }
        if self.topk % 64 != 0 {
            return Err(Error::InvalidArgument(format!(
                "SM90 sparse decode requires topk to be a multiple of 64, got {}",
                self.topk
            )));
        }
        if self.has_extra_cache() {
            if self.extra_num_blocks == 0 || self.extra_page_block_size == 0 || self.extra_topk == 0
            {
                return Err(Error::InvalidArgument(
                    "extra KV cache requires non-zero extra_num_blocks, extra_page_block_size, and extra_topk"
                        .to_string(),
                ));
            }
            if self.d_qk == 576 {
                return Err(Error::InvalidArgument(
                    "V32 sparse decode does not support extra KV cache".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Returns the required FlashMLA packed FP8 KV-cache bytes per token.
    pub fn kv_bytes_per_token(self) -> Result<usize> {
        match (self.d_qk, self.d_v) {
            (576, 512) => Ok(656),
            (512, 512) => Ok(584),
            _ => Err(Error::InvalidArgument(format!(
                "unsupported sparse decode KV layout for d_qk={} d_v={}",
                self.d_qk, self.d_v
            ))),
        }
    }

    /// Returns true when optional extra KV-cache tensors are part of the decode launch.
    pub fn has_extra_cache(self) -> bool {
        self.extra_num_blocks != 0 || self.extra_page_block_size != 0 || self.extra_topk != 0
    }
}

/// Workspace and scheduler sizing returned by sparse decode planning.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseDecodePlanMeta {
    /// Number of SM partitions used by split-KV decode.
    pub num_sm_parts: usize,
    /// Fixed scheduler overhead, in top-k blocks.
    pub fixed_overhead_num_blocks: usize,
    /// Top-k block size used by the scheduler.
    pub block_size_topk: usize,
    /// Required I32 elements in scheduler metadata.
    pub scheduler_metadata_i32_len: usize,
    /// Required I32 elements in split offsets.
    pub num_splits_len: usize,
    /// Required F32 elements in LSE accumulation workspace.
    pub lse_accum_elem_count: usize,
    /// Required F32 elements in output accumulation workspace.
    pub o_accum_elem_count: usize,
}

impl SparseDecodePlanMeta {
    /// Converts the raw FFI plan result into Rust workspace metadata.
    pub fn from_sys(result: flashmla_sparse_decode_plan_result_t) -> Result<Self> {
        Ok(Self {
            num_sm_parts: checked_usize(result.num_sm_parts, "num_sm_parts")?,
            fixed_overhead_num_blocks: checked_usize(
                result.fixed_overhead_num_blocks,
                "fixed_overhead_num_blocks",
            )?,
            block_size_topk: checked_usize(result.block_size_topk, "block_size_topk")?,
            scheduler_metadata_i32_len: result.scheduler_metadata_i32_len,
            num_splits_len: result.num_splits_len,
            lse_accum_elem_count: result.lse_accum_elem_count,
            o_accum_elem_count: result.o_accum_elem_count,
        })
    }

    /// Returns the total number of split rows in decode accumulation workspaces.
    pub fn total_num_splits(self, dims: SparseDecodeDims) -> Result<usize> {
        dims.batch
            .checked_add(self.num_sm_parts)
            .ok_or_else(|| Error::InvalidArgument("total_num_splits overflow".to_string()))
    }
}

/// Raw pointer parameters for sparse decode planning and scheduler metadata generation.
#[derive(Debug, Copy, Clone)]
pub struct SparseDecodePlanParams {
    /// Sparse decode tensor dimensions.
    pub dims: SparseDecodeDims,
    /// Optional raw I32 top-k length pointer shaped `[batch]`.
    pub topk_length: *const i32,
    /// Optional raw I32 extra top-k length pointer shaped `[batch]`.
    pub extra_topk_length: *const i32,
    /// Optional writable I32 scheduler metadata buffer.
    pub tile_scheduler_metadata: *mut i32,
    /// Optional writable I32 split-offset buffer.
    pub num_splits: *mut i32,
    /// Number of SMs on the target CUDA device.
    pub num_sm: usize,
    /// CUDA stream used for optional metadata generation.
    pub stream: cudaStream_t,
}

impl SparseDecodePlanParams {
    fn validate(self) -> Result<()> {
        self.dims.validate()?;
        if self.num_sm == 0 {
            return Err(Error::InvalidArgument(
                "num_sm must be non-zero".to_string(),
            ));
        }
        if self.dims.d_qk == 576 && !self.topk_length.is_null() {
            return Err(Error::InvalidArgument(
                "V32 sparse decode does not support topk_length".to_string(),
            ));
        }
        if !self.dims.has_extra_cache() && !self.extra_topk_length.is_null() {
            return Err(Error::InvalidArgument(
                "extra_topk_length requires an extra KV cache".to_string(),
            ));
        }
        if self.tile_scheduler_metadata.is_null() != self.num_splits.is_null() {
            return Err(Error::InvalidArgument(
                "tile_scheduler_metadata and num_splits must both be null or both be non-null"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn to_sys(self) -> Result<flashmla_sparse_decode_plan_params_t> {
        self.validate()?;
        Ok(flashmla_sparse_decode_plan_params_t {
            batch: checked_i32(self.dims.batch, "batch")?,
            s_q: checked_i32(self.dims.s_q, "s_q")?,
            h_q: checked_i32(self.dims.h_q, "h_q")?,
            h_kv: checked_i32(self.dims.h_kv, "h_kv")?,
            d_qk: checked_i32(self.dims.d_qk, "d_qk")?,
            d_v: checked_i32(self.dims.d_v, "d_v")?,
            topk: checked_i32(self.dims.topk, "topk")?,
            extra_topk: checked_i32(self.dims.extra_topk, "extra_topk")?,
            topk_length: self.topk_length,
            extra_topk_length: self.extra_topk_length,
            tile_scheduler_metadata: self.tile_scheduler_metadata,
            num_splits: self.num_splits,
            num_sm: checked_i32(self.num_sm, "num_sm")?,
            stream: self.stream,
        })
    }
}

/// Element strides for sparse decode tensors and workspaces.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseDecodeStrides {
    /// Element stride between query batches.
    pub q_b: usize,
    /// Element stride between query positions.
    pub q_s_q: usize,
    /// Element stride between query heads.
    pub q_h_q: usize,
    /// Byte stride between packed KV cache pages.
    pub kv_block: usize,
    /// Byte stride between packed KV cache rows.
    pub kv_row: usize,
    /// Element stride between sparse-index batches.
    pub indices_b: usize,
    /// Element stride between sparse-index positions.
    pub indices_s_q: usize,
    /// Element stride between LSE batches.
    pub lse_b: usize,
    /// Element stride between LSE positions.
    pub lse_s_q: usize,
    /// Element stride between output batches.
    pub out_b: usize,
    /// Element stride between output positions.
    pub out_s_q: usize,
    /// Element stride between output heads.
    pub out_h_q: usize,
    /// Byte stride between optional extra KV cache pages.
    pub extra_kv_block: usize,
    /// Byte stride between optional extra KV cache rows.
    pub extra_kv_row: usize,
    /// Element stride between optional extra sparse-index batches.
    pub extra_indices_b: usize,
    /// Element stride between optional extra sparse-index positions.
    pub extra_indices_s_q: usize,
    /// Element stride between LSE accumulation splits.
    pub lse_accum_split: usize,
    /// Element stride between LSE accumulation positions.
    pub lse_accum_s_q: usize,
    /// Element stride between output accumulation splits.
    pub o_accum_split: usize,
    /// Element stride between output accumulation positions.
    pub o_accum_s_q: usize,
    /// Element stride between output accumulation heads.
    pub o_accum_h_q: usize,
}

impl SparseDecodeStrides {
    /// Validates that required sparse decode strides are positive and fit in the C ABI.
    pub fn validate(self, has_extra_cache: bool) -> Result<()> {
        checked_stride(self.q_b, "q_b")?;
        checked_stride(self.q_s_q, "q_s_q")?;
        checked_stride(self.q_h_q, "q_h_q")?;
        checked_stride(self.kv_block, "kv_block")?;
        checked_stride(self.kv_row, "kv_row")?;
        checked_stride(self.indices_b, "indices_b")?;
        checked_stride(self.indices_s_q, "indices_s_q")?;
        checked_stride(self.lse_b, "lse_b")?;
        checked_stride(self.lse_s_q, "lse_s_q")?;
        checked_stride(self.out_b, "out_b")?;
        checked_stride(self.out_s_q, "out_s_q")?;
        checked_stride(self.out_h_q, "out_h_q")?;
        checked_stride(self.lse_accum_split, "lse_accum_split")?;
        checked_stride(self.lse_accum_s_q, "lse_accum_s_q")?;
        checked_stride(self.o_accum_split, "o_accum_split")?;
        checked_stride(self.o_accum_s_q, "o_accum_s_q")?;
        checked_stride(self.o_accum_h_q, "o_accum_h_q")?;
        if has_extra_cache {
            checked_stride(self.extra_kv_block, "extra_kv_block")?;
            checked_stride(self.extra_kv_row, "extra_kv_row")?;
            checked_stride(self.extra_indices_b, "extra_indices_b")?;
            checked_stride(self.extra_indices_s_q, "extra_indices_s_q")?;
        }
        Ok(())
    }
}

/// Raw pointer parameters for launching sparse BF16-query / FP8-cache decode.
#[derive(Debug, Copy, Clone)]
pub struct SparseDecodeLaunchParams {
    /// Sparse decode tensor dimensions.
    pub dims: SparseDecodeDims,
    /// Sparse decode runtime options.
    pub config: SparseDecodeConfig,
    /// Raw BF16 query pointer with shape `[batch, s_q, h_q, d_qk]`.
    pub q: *const c_void,
    /// Raw packed FP8 KV cache pointer.
    pub kv: *const c_void,
    /// Raw I32 sparse index pointer with shape `[batch, s_q, topk]`.
    pub indices: *const i32,
    /// Optional raw I32 top-k length pointer with shape `[batch]`.
    pub topk_length: *const i32,
    /// Optional raw F32 attention sink pointer with shape `[h_q]`.
    pub attn_sink: *const f32,
    /// Raw BF16 output pointer with shape `[batch, s_q, h_q, d_v]`.
    pub out: *mut c_void,
    /// Raw F32 LSE output pointer with shape `[batch, s_q, h_q]`.
    pub lse: *mut f32,
    /// Optional raw packed FP8 extra KV cache pointer.
    pub extra_kv: *const c_void,
    /// Optional raw I32 extra sparse index pointer with shape `[batch, s_q, extra_topk]`.
    pub extra_indices: *const i32,
    /// Optional raw I32 extra top-k length pointer with shape `[batch]`.
    pub extra_topk_length: *const i32,
    /// Element strides for all tensors and workspaces.
    pub strides: SparseDecodeStrides,
    /// Raw F32 LSE accumulation workspace pointer.
    pub lse_accum: *mut f32,
    /// Raw F32 output accumulation workspace pointer.
    pub o_accum: *mut f32,
    /// Raw I32 scheduler metadata pointer generated by sparse decode planning.
    pub tile_scheduler_metadata: *mut i32,
    /// Raw I32 split-offset pointer generated by sparse decode planning.
    pub num_splits: *mut i32,
    /// Number of SM partitions from sparse decode planning.
    pub num_sm_parts: usize,
    /// CUDA stream used for decode and combine launches.
    pub stream: cudaStream_t,
}

impl SparseDecodeLaunchParams {
    fn validate(self) -> Result<()> {
        self.dims.validate()?;
        self.strides.validate(self.dims.has_extra_cache())?;
        if self.config.d_v != self.dims.d_v {
            return Err(Error::InvalidArgument(format!(
                "config d_v ({}) must match dims d_v ({})",
                self.config.d_v, self.dims.d_v
            )));
        }
        if self.q.is_null()
            || self.kv.is_null()
            || self.indices.is_null()
            || self.out.is_null()
            || self.lse.is_null()
            || self.lse_accum.is_null()
            || self.o_accum.is_null()
            || self.tile_scheduler_metadata.is_null()
            || self.num_splits.is_null()
        {
            return Err(Error::InvalidArgument(
                "q, kv, indices, out, lse, lse_accum, o_accum, tile_scheduler_metadata, and num_splits pointers must be non-null"
                    .to_string(),
            ));
        }
        if self.num_sm_parts == 0 {
            return Err(Error::InvalidArgument(
                "num_sm_parts must be non-zero".to_string(),
            ));
        }
        if self.dims.d_qk == 576 && !self.topk_length.is_null() {
            return Err(Error::InvalidArgument(
                "V32 sparse decode does not support topk_length".to_string(),
            ));
        }
        if self.dims.has_extra_cache() {
            if self.extra_kv.is_null() || self.extra_indices.is_null() {
                return Err(Error::InvalidArgument(
                    "extra KV cache requires extra_kv and extra_indices pointers".to_string(),
                ));
            }
        } else if !self.extra_kv.is_null()
            || !self.extra_indices.is_null()
            || !self.extra_topk_length.is_null()
        {
            return Err(Error::InvalidArgument(
                "extra pointers require extra KV cache dimensions".to_string(),
            ));
        }
        Ok(())
    }

    fn to_sys(self) -> Result<flashmla_sparse_decode_params_t> {
        self.validate()?;
        Ok(flashmla_sparse_decode_params_t {
            batch: checked_i32(self.dims.batch, "batch")?,
            s_q: checked_i32(self.dims.s_q, "s_q")?,
            h_q: checked_i32(self.dims.h_q, "h_q")?,
            h_kv: checked_i32(self.dims.h_kv, "h_kv")?,
            d_qk: checked_i32(self.dims.d_qk, "d_qk")?,
            d_v: checked_i32(self.dims.d_v, "d_v")?,
            num_blocks: checked_i32(self.dims.num_blocks, "num_blocks")?,
            page_block_size: checked_i32(self.dims.page_block_size, "page_block_size")?,
            topk: checked_i32(self.dims.topk, "topk")?,
            sm_scale: self.config.softmax_scale,
            q: self.q,
            kv: self.kv,
            indices: self.indices,
            topk_length: self.topk_length,
            attn_sink: self.attn_sink,
            out: self.out,
            lse: self.lse,
            extra_num_blocks: checked_i32(self.dims.extra_num_blocks, "extra_num_blocks")?,
            extra_page_block_size: checked_i32(
                self.dims.extra_page_block_size,
                "extra_page_block_size",
            )?,
            extra_topk: checked_i32(self.dims.extra_topk, "extra_topk")?,
            extra_kv: self.extra_kv,
            extra_indices: self.extra_indices,
            extra_topk_length: self.extra_topk_length,
            stride_q_b: checked_stride(self.strides.q_b, "q_b")?,
            stride_q_s_q: checked_stride(self.strides.q_s_q, "q_s_q")?,
            stride_q_h_q: checked_stride(self.strides.q_h_q, "q_h_q")?,
            stride_kv_block: checked_stride(self.strides.kv_block, "kv_block")?,
            stride_kv_row: checked_stride(self.strides.kv_row, "kv_row")?,
            stride_indices_b: checked_stride(self.strides.indices_b, "indices_b")?,
            stride_indices_s_q: checked_stride(self.strides.indices_s_q, "indices_s_q")?,
            stride_lse_b: checked_stride(self.strides.lse_b, "lse_b")?,
            stride_lse_s_q: checked_stride(self.strides.lse_s_q, "lse_s_q")?,
            stride_o_b: checked_stride(self.strides.out_b, "out_b")?,
            stride_o_s_q: checked_stride(self.strides.out_s_q, "out_s_q")?,
            stride_o_h_q: checked_stride(self.strides.out_h_q, "out_h_q")?,
            stride_extra_kv_block: checked_optional_stride(
                self.strides.extra_kv_block,
                "extra_kv_block",
                self.dims.has_extra_cache(),
            )?,
            stride_extra_kv_row: checked_optional_stride(
                self.strides.extra_kv_row,
                "extra_kv_row",
                self.dims.has_extra_cache(),
            )?,
            stride_extra_indices_b: checked_optional_stride(
                self.strides.extra_indices_b,
                "extra_indices_b",
                self.dims.has_extra_cache(),
            )?,
            stride_extra_indices_s_q: checked_optional_stride(
                self.strides.extra_indices_s_q,
                "extra_indices_s_q",
                self.dims.has_extra_cache(),
            )?,
            lse_accum: self.lse_accum,
            o_accum: self.o_accum,
            stride_lse_accum_split: checked_stride(
                self.strides.lse_accum_split,
                "lse_accum_split",
            )?,
            stride_lse_accum_s_q: checked_stride(self.strides.lse_accum_s_q, "lse_accum_s_q")?,
            stride_o_accum_split: checked_stride(self.strides.o_accum_split, "o_accum_split")?,
            stride_o_accum_s_q: checked_stride(self.strides.o_accum_s_q, "o_accum_s_q")?,
            stride_o_accum_h_q: checked_stride(self.strides.o_accum_h_q, "o_accum_h_q")?,
            tile_scheduler_metadata: self.tile_scheduler_metadata,
            num_splits: self.num_splits,
            num_sm_parts: checked_i32(self.num_sm_parts, "num_sm_parts")?,
            stream: self.stream,
        })
    }
}

/// Computes sparse decode workspace metadata and optionally generates scheduler metadata.
///
/// # Safety
///
/// If `params.tile_scheduler_metadata` and `params.num_splits` are non-null, they must be valid
/// writable CUDA device pointers sized according to the metadata returned by a size-only planning
/// call with the same dimensions. Optional length pointers must be valid CUDA device pointers for
/// the documented shapes. `params.stream` must be a valid CUDA stream for the current device, or
/// null for the default stream.
pub unsafe fn sparse_decode_plan(params: &SparseDecodePlanParams) -> Result<SparseDecodePlanMeta> {
    let sys_params = params.to_sys()?;
    let mut result = flashmla_sparse_decode_plan_result_t {
        num_sm_parts: 0,
        fixed_overhead_num_blocks: 0,
        block_size_topk: 0,
        scheduler_metadata_i32_len: 0,
        num_splits_len: 0,
        lse_accum_elem_count: 0,
        o_accum_elem_count: 0,
    };
    let status = unsafe { sys_sparse_decode_plan(&sys_params, &mut result) };
    if status == flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        SparseDecodePlanMeta::from_sys(result)
    } else {
        Err(Error::from_status(status, "sparse decode planning failed"))
    }
}

/// Launches SM90 sparse decode and combine through `flashmla-sys`.
///
/// # Safety
///
/// All raw pointers in `params` must be valid CUDA device pointers for the documented shapes,
/// dtypes, and element strides. Workspace and scheduler buffers must come from
/// `sparse_decode_plan` for identical dimensions and top-k lengths. Output buffers must be
/// writable and must not alias inputs in a way that violates upstream FlashMLA kernel
/// requirements. `params.stream` must be a valid CUDA stream for the current device, or null for
/// the default stream.
pub unsafe fn sparse_decode_bf16_fp8(params: &SparseDecodeLaunchParams) -> Result<()> {
    let sys_params = params.to_sys()?;
    let status = unsafe { sys_sparse_decode_bf16_fp8(&sys_params) };
    if status == flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_status(status, "sparse decode launch failed"))
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} does not fit in i32: {value}")))
}

fn checked_usize(value: i32, name: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} is negative: {value}")))
}

fn checked_stride(value: usize, name: &str) -> Result<i32> {
    if value == 0 {
        return Err(Error::InvalidArgument(format!("{name} must be non-zero")));
    }
    checked_i32(value, name)
}

fn checked_optional_stride(value: usize, name: &str, required: bool) -> Result<i32> {
    if required {
        checked_stride(value, name)
    } else {
        checked_i32(value, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_supported_shape() {
        SparseDecodeDims {
            batch: 1,
            s_q: 1,
            h_q: 64,
            h_kv: 1,
            d_qk: 576,
            d_v: 512,
            num_blocks: 2,
            page_block_size: 64,
            topk: 64,
            extra_num_blocks: 0,
            extra_page_block_size: 0,
            extra_topk: 0,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn rejects_v32_extra_cache() {
        let dims = SparseDecodeDims {
            batch: 1,
            s_q: 1,
            h_q: 64,
            h_kv: 1,
            d_qk: 576,
            d_v: 512,
            num_blocks: 2,
            page_block_size: 64,
            topk: 64,
            extra_num_blocks: 1,
            extra_page_block_size: 64,
            extra_topk: 64,
        };
        assert!(matches!(dims.validate(), Err(Error::InvalidArgument(_))));
    }
}

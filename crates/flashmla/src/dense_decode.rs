use std::ffi::c_void;

use flashmla_sys::{
    cudaStream_t, flashmla_dense_decode_bf16 as sys_dense_decode_bf16,
    flashmla_dense_decode_params_t, flashmla_dense_decode_plan as sys_dense_decode_plan,
    flashmla_dense_decode_plan_params_t, flashmla_dense_decode_plan_result_t, flashmla_status_t,
};

use crate::{Error, Result};

/// Runtime options for dense decode.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DenseDecodeConfig {
    /// Softmax scale applied to QK logits.
    pub softmax_scale: f32,
    /// Value head dimension. FlashMLA dense decode currently requires `512`.
    pub d_v: usize,
    /// Whether dense decode should apply a causal mask. Ignored when `s_q == 1`.
    pub is_causal: bool,
}

impl Default for DenseDecodeConfig {
    fn default() -> Self {
        Self {
            softmax_scale: 1.0,
            d_v: 512,
            is_causal: false,
        }
    }
}

/// Shape parameters for dense decode.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DenseDecodeDims {
    /// Batch size.
    pub batch: usize,
    /// Query sequence length before query-head packing.
    pub s_q: usize,
    /// Query head count.
    pub h_q: usize,
    /// KV head count.
    pub h_k: usize,
    /// Query/key head dimension. FlashMLA supports `512` and `576`.
    pub d_qk: usize,
    /// Value head dimension. FlashMLA dense decode currently requires `512`.
    pub d_v: usize,
    /// Number of pages in the BF16 KV cache.
    pub num_blocks: usize,
    /// Number of tokens in each KV cache page. FlashMLA dense decode currently requires `64`.
    pub page_block_size: usize,
}

impl DenseDecodeDims {
    /// Validates architecture-independent dense decode shape constraints.
    pub fn validate(self) -> Result<()> {
        if self.batch == 0 || self.s_q == 0 || self.h_q == 0 || self.h_k == 0 {
            return Err(Error::InvalidArgument(
                "batch, s_q, h_q, and h_k must be non-zero".to_string(),
            ));
        }
        if self.num_blocks == 0 || self.page_block_size == 0 {
            return Err(Error::InvalidArgument(
                "num_blocks and page_block_size must be non-zero".to_string(),
            ));
        }
        if self.page_block_size != 64 {
            return Err(Error::InvalidArgument(format!(
                "dense decode requires page_block_size to be 64, got {}",
                self.page_block_size
            )));
        }
        if self.h_q % self.h_k != 0 {
            return Err(Error::InvalidArgument(format!(
                "h_k ({}) must divide h_q ({})",
                self.h_k, self.h_q
            )));
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
        self.q_seq_per_hk()?;
        Ok(())
    }

    /// Returns the number of query heads grouped under each KV head.
    pub fn q_heads_per_hk(self) -> Result<usize> {
        if self.h_k == 0 || self.h_q % self.h_k != 0 {
            return Err(Error::InvalidArgument(
                "h_k must be non-zero and divide h_q".to_string(),
            ));
        }
        Ok(self.h_q / self.h_k)
    }

    /// Returns the packed query row count per KV head.
    pub fn q_seq_per_hk(self) -> Result<usize> {
        self.s_q
            .checked_mul(self.q_heads_per_hk()?)
            .ok_or_else(|| Error::InvalidArgument("q_seq_per_hk overflow".to_string()))
    }
}

/// Workspace and scheduler sizing returned by dense decode planning.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DenseDecodePlanMeta {
    /// Number of SM partitions used by split-KV decode.
    pub num_sm_parts: usize,
    /// Fixed scheduler overhead, in KV blocks.
    pub fixed_overhead_num_blocks: usize,
    /// KV block size used by the scheduler.
    pub block_size_n: usize,
    /// Packed query sequence length per KV head.
    pub q_seq_per_hk: usize,
    /// Required I32 elements in scheduler metadata.
    pub scheduler_metadata_i32_len: usize,
    /// Required I32 elements in split offsets.
    pub num_splits_len: usize,
    /// Required F32 elements in dense internal LSE accumulation workspace.
    pub lse_accum_elem_count: usize,
    /// Required F32 elements in dense internal output accumulation workspace.
    pub o_accum_elem_count: usize,
}

impl DenseDecodePlanMeta {
    /// Converts the raw FFI plan result into Rust workspace metadata.
    pub fn from_sys(result: flashmla_dense_decode_plan_result_t) -> Result<Self> {
        Ok(Self {
            num_sm_parts: checked_usize(result.num_sm_parts, "num_sm_parts")?,
            fixed_overhead_num_blocks: checked_usize(
                result.fixed_overhead_num_blocks,
                "fixed_overhead_num_blocks",
            )?,
            block_size_n: checked_usize(result.block_size_n, "block_size_n")?,
            q_seq_per_hk: checked_usize(result.q_seq_per_hk, "q_seq_per_hk")?,
            scheduler_metadata_i32_len: result.scheduler_metadata_i32_len,
            num_splits_len: result.num_splits_len,
            lse_accum_elem_count: result.lse_accum_elem_count,
            o_accum_elem_count: result.o_accum_elem_count,
        })
    }

    /// Returns the total number of split rows in dense decode accumulation workspaces.
    pub fn total_num_splits(self, dims: DenseDecodeDims) -> Result<usize> {
        dims.batch
            .checked_add(self.num_sm_parts)
            .ok_or_else(|| Error::InvalidArgument("total_num_splits overflow".to_string()))
    }
}

/// Raw pointer parameters for dense decode planning and scheduler metadata generation.
#[derive(Debug, Copy, Clone)]
pub struct DenseDecodePlanParams {
    /// Dense decode tensor dimensions.
    pub dims: DenseDecodeDims,
    /// Optional raw I32 KV sequence lengths pointer shaped `[batch]`.
    pub seqlens_k: *const i32,
    /// Optional writable I32 scheduler metadata buffer.
    pub tile_scheduler_metadata: *mut i32,
    /// Optional writable I32 split-offset buffer.
    pub num_splits: *mut i32,
    /// Number of SMs on the target CUDA device.
    pub num_sm: usize,
    /// CUDA stream used for optional metadata generation.
    pub stream: cudaStream_t,
}

impl DenseDecodePlanParams {
    fn validate(self) -> Result<()> {
        self.dims.validate()?;
        if self.num_sm == 0 {
            return Err(Error::InvalidArgument(
                "num_sm must be non-zero".to_string(),
            ));
        }
        if self.tile_scheduler_metadata.is_null() != self.num_splits.is_null() {
            return Err(Error::InvalidArgument(
                "tile_scheduler_metadata and num_splits must both be null or both be non-null"
                    .to_string(),
            ));
        }
        if !self.tile_scheduler_metadata.is_null() && self.seqlens_k.is_null() {
            return Err(Error::InvalidArgument(
                "seqlens_k must be non-null when generating dense decode metadata".to_string(),
            ));
        }
        Ok(())
    }

    fn to_sys(self) -> Result<flashmla_dense_decode_plan_params_t> {
        self.validate()?;
        Ok(flashmla_dense_decode_plan_params_t {
            batch: checked_i32(self.dims.batch, "batch")?,
            s_q: checked_i32(self.dims.s_q, "s_q")?,
            h_q: checked_i32(self.dims.h_q, "h_q")?,
            h_k: checked_i32(self.dims.h_k, "h_k")?,
            d_qk: checked_i32(self.dims.d_qk, "d_qk")?,
            d_v: checked_i32(self.dims.d_v, "d_v")?,
            seqlens_k: self.seqlens_k,
            tile_scheduler_metadata: self.tile_scheduler_metadata,
            num_splits: self.num_splits,
            num_sm: checked_i32(self.num_sm, "num_sm")?,
            stream: self.stream,
        })
    }
}

/// Element strides for raw dense decode tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DenseDecodeStrides {
    /// Element stride between packed query batches.
    pub q_b: usize,
    /// Element stride between packed query rows.
    pub q_row: usize,
    /// Element stride between packed query KV heads.
    pub q_head: usize,
    /// Element stride between KV cache pages.
    pub k_block: usize,
    /// Element stride between KV cache rows.
    pub k_row: usize,
    /// Element stride between KV cache heads.
    pub k_head: usize,
    /// Element stride between block-table batches.
    pub block_table_b: usize,
}

impl DenseDecodeStrides {
    /// Validates that all raw tensor strides are positive and fit in the C ABI.
    pub fn validate(self) -> Result<()> {
        checked_stride(self.q_b, "q_b")?;
        checked_stride(self.q_row, "q_row")?;
        checked_stride(self.q_head, "q_head")?;
        checked_stride(self.k_block, "k_block")?;
        checked_stride(self.k_row, "k_row")?;
        checked_stride(self.k_head, "k_head")?;
        checked_stride(self.block_table_b, "block_table_b")?;
        Ok(())
    }
}

/// Raw pointer parameters for launching BF16 dense decode.
#[derive(Debug, Copy, Clone)]
pub struct DenseDecodeLaunchParams {
    /// Dense decode tensor dimensions.
    pub dims: DenseDecodeDims,
    /// Dense decode runtime options.
    pub config: DenseDecodeConfig,
    /// Raw BF16 packed query pointer with shape `[batch, q_seq_per_hk, h_k, d_qk]`.
    pub q: *const c_void,
    /// Raw BF16 KV cache pointer with shape `[num_blocks, page_block_size, h_k, d_qk]`.
    pub kcache: *const c_void,
    /// Raw I32 KV sequence lengths pointer with shape `[batch]`.
    pub seqlens_k: *const i32,
    /// Raw I32 block table pointer with shape `[batch, max_num_blocks_per_seq]`.
    pub block_table: *const i32,
    /// Raw BF16 internal output pointer with shape `[batch, h_k, q_seq_per_hk, d_v]`.
    pub out: *mut c_void,
    /// Raw F32 internal LSE pointer with shape `[batch, h_k, q_seq_per_hk]`.
    pub lse: *mut f32,
    /// Raw F32 LSE accumulation workspace pointer.
    pub lse_accum: *mut f32,
    /// Raw F32 output accumulation workspace pointer.
    pub o_accum: *mut f32,
    /// Element strides for packed query, KV cache, and block table.
    pub strides: DenseDecodeStrides,
    /// Raw I32 scheduler metadata pointer generated by dense decode planning.
    pub tile_scheduler_metadata: *mut i32,
    /// Raw I32 split-offset pointer generated by dense decode planning.
    pub num_splits: *mut i32,
    /// Number of SM partitions from dense decode planning.
    pub num_sm_parts: usize,
    /// CUDA stream used for decode and combine launches.
    pub stream: cudaStream_t,
}

impl DenseDecodeLaunchParams {
    fn validate(self) -> Result<()> {
        self.dims.validate()?;
        self.strides.validate()?;
        if self.config.d_v != self.dims.d_v {
            return Err(Error::InvalidArgument(format!(
                "config d_v ({}) must match dims d_v ({})",
                self.config.d_v, self.dims.d_v
            )));
        }
        if self.q.is_null()
            || self.kcache.is_null()
            || self.seqlens_k.is_null()
            || self.block_table.is_null()
            || self.out.is_null()
            || self.lse.is_null()
            || self.lse_accum.is_null()
            || self.o_accum.is_null()
            || self.tile_scheduler_metadata.is_null()
            || self.num_splits.is_null()
        {
            return Err(Error::InvalidArgument(
                "q, kcache, seqlens_k, block_table, out, lse, lse_accum, o_accum, tile_scheduler_metadata, and num_splits pointers must be non-null"
                    .to_string(),
            ));
        }
        if self.num_sm_parts == 0 {
            return Err(Error::InvalidArgument(
                "num_sm_parts must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    fn to_sys(self) -> Result<flashmla_dense_decode_params_t> {
        self.validate()?;
        Ok(flashmla_dense_decode_params_t {
            batch: checked_i32(self.dims.batch, "batch")?,
            s_q: checked_i32(self.dims.s_q, "s_q")?,
            h_q: checked_i32(self.dims.h_q, "h_q")?,
            h_k: checked_i32(self.dims.h_k, "h_k")?,
            d_qk: checked_i32(self.dims.d_qk, "d_qk")?,
            d_v: checked_i32(self.dims.d_v, "d_v")?,
            num_blocks: checked_i32(self.dims.num_blocks, "num_blocks")?,
            page_block_size: checked_i32(self.dims.page_block_size, "page_block_size")?,
            is_causal: if self.config.is_causal { 1 } else { 0 },
            sm_scale: self.config.softmax_scale,
            q: self.q,
            kcache: self.kcache,
            seqlens_k: self.seqlens_k,
            block_table: self.block_table,
            out: self.out,
            lse: self.lse,
            lse_accum: self.lse_accum,
            o_accum: self.o_accum,
            stride_q_b: checked_stride(self.strides.q_b, "q_b")?,
            stride_q_row: checked_stride(self.strides.q_row, "q_row")?,
            stride_q_head: checked_stride(self.strides.q_head, "q_head")?,
            stride_k_block: checked_stride(self.strides.k_block, "k_block")?,
            stride_k_row: checked_stride(self.strides.k_row, "k_row")?,
            stride_k_head: checked_stride(self.strides.k_head, "k_head")?,
            stride_block_table_b: checked_stride(self.strides.block_table_b, "block_table_b")?,
            tile_scheduler_metadata: self.tile_scheduler_metadata,
            num_splits: self.num_splits,
            num_sm_parts: checked_i32(self.num_sm_parts, "num_sm_parts")?,
            stream: self.stream,
        })
    }
}

/// Computes dense decode workspace metadata and optionally generates scheduler metadata.
///
/// # Safety
///
/// If `params.tile_scheduler_metadata` and `params.num_splits` are non-null, they must be valid
/// writable CUDA device pointers sized according to the metadata returned by a size-only planning
/// call with the same dimensions. In that mode `params.seqlens_k` must be a valid CUDA device
/// pointer shaped `[batch]`. `params.stream` must be a valid CUDA stream for the current device,
/// or null for the default stream.
pub unsafe fn dense_decode_plan(params: &DenseDecodePlanParams) -> Result<DenseDecodePlanMeta> {
    let sys_params = params.to_sys()?;
    let mut result = flashmla_dense_decode_plan_result_t {
        num_sm_parts: 0,
        fixed_overhead_num_blocks: 0,
        block_size_n: 0,
        q_seq_per_hk: 0,
        scheduler_metadata_i32_len: 0,
        num_splits_len: 0,
        lse_accum_elem_count: 0,
        o_accum_elem_count: 0,
    };
    let status = unsafe { sys_dense_decode_plan(&sys_params, &mut result) };
    if status == flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        DenseDecodePlanMeta::from_sys(result)
    } else {
        Err(Error::from_status(status, "dense decode planning failed"))
    }
}

/// Launches SM90 BF16 dense decode and combine through `flashmla-sys`.
///
/// # Safety
///
/// All raw pointers in `params` must be valid CUDA device pointers for the documented shapes,
/// dtypes, and element strides. Workspace and scheduler buffers must come from
/// `dense_decode_plan` for identical dimensions and sequence lengths. Output buffers must be
/// writable and must not alias inputs in a way that violates upstream FlashMLA kernel
/// requirements. `params.stream` must be a valid CUDA stream for the current device, or null for
/// the default stream.
pub unsafe fn dense_decode_bf16(params: &DenseDecodeLaunchParams) -> Result<()> {
    let sys_params = params.to_sys()?;
    let status = unsafe { sys_dense_decode_bf16(&sys_params) };
    if status == flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_status(status, "dense decode launch failed"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_supported_shape() {
        DenseDecodeDims {
            batch: 1,
            s_q: 2,
            h_q: 8,
            h_k: 2,
            d_qk: 576,
            d_v: 512,
            num_blocks: 2,
            page_block_size: 64,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn rejects_non_divisible_heads() {
        let dims = DenseDecodeDims {
            batch: 1,
            s_q: 2,
            h_q: 7,
            h_k: 2,
            d_qk: 576,
            d_v: 512,
            num_blocks: 2,
            page_block_size: 64,
        };
        assert!(matches!(dims.validate(), Err(Error::InvalidArgument(_))));
    }
}

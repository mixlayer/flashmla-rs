//! Dense decode API re-exports and Candle CUDA integration.

pub use flashmla::{DenseDecodeConfig, DenseDecodeDims, DenseDecodePlanMeta};

use candle::{DType, Tensor};
use flashmla::{
    DenseDecodeLaunchParams, DenseDecodePlanParams, DenseDecodeStrides, dense_decode_bf16,
    dense_decode_plan as flashmla_dense_decode_plan, get_device_info,
};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_contiguous, ensure_dtype, ensure_last_dim_contiguous, ensure_rank,
        ensure_same_device, stream_and_device_id, tensor_ptr_bf16, tensor_ptr_f32, tensor_ptr_i32,
    },
};

/// Caller-owned dense decode scheduler and split-KV workspaces.
#[derive(Debug)]
pub struct DenseDecodePlan {
    /// I32 scheduler metadata shaped `[num_sm_parts, metadata_i32_per_part]`.
    pub scheduler_metadata: Tensor,
    /// I32 split offsets shaped `[batch + 1]`.
    pub num_splits: Tensor,
    /// F32 split-KV LSE accumulation workspace shaped `[batch + num_sm_parts, h_k, q_seq_per_hk]`.
    pub lse_accum: Tensor,
    /// F32 split-KV output accumulation workspace shaped `[batch + num_sm_parts, h_k, q_seq_per_hk, d_v]`.
    pub o_accum: Tensor,
    /// Workspace sizing metadata returned by FlashMLA planning.
    pub meta: DenseDecodePlanMeta,
}

/// Output tensors returned by dense decode.
#[derive(Debug)]
pub struct DenseDecodeOutput {
    /// BF16 attention output shaped `[batch, s_q, h_q, d_v]`.
    pub out: Tensor,
    /// F32 log-sum-exp values shaped `[batch, h_q, s_q]` to mirror upstream FlashMLA.
    pub lse: Tensor,
}

/// Allocates dense decode scheduler metadata and split-KV workspaces.
pub fn dense_decode_plan(
    q: &Tensor,
    k_cache: &Tensor,
    seqlens_k: &Tensor,
    config: DenseDecodeConfig,
) -> Result<DenseDecodePlan> {
    let dims = validate_decode_tensors(q, k_cache, seqlens_k, None, config)?;
    let (stream, device_id) = stream_and_device_id(q)?;
    let device = get_device_info(device_id)?;
    let num_sm = usize::try_from(device.num_sms)
        .map_err(|_| crate::Error::Tensor("num_sms overflow".to_string()))?;

    let meta = {
        let plan_params = DenseDecodePlanParams {
            dims,
            seqlens_k: std::ptr::null(),
            tile_scheduler_metadata: std::ptr::null_mut(),
            num_splits: std::ptr::null_mut(),
            num_sm,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        unsafe { flashmla_dense_decode_plan(&plan_params)? }
    };

    if meta.num_sm_parts == 0 || meta.scheduler_metadata_i32_len % meta.num_sm_parts != 0 {
        return invalid_arg("invalid dense decode scheduler metadata shape");
    }
    if meta.q_seq_per_hk != dims.q_seq_per_hk()? {
        return invalid_arg("dense decode plan q_seq_per_hk does not match tensor dimensions");
    }

    let metadata_i32_per_part = meta.scheduler_metadata_i32_len / meta.num_sm_parts;
    let total_num_splits = meta.total_num_splits(dims)?;
    let q_seq_per_hk = dims.q_seq_per_hk()?;

    let scheduler_metadata = Tensor::zeros(
        (meta.num_sm_parts, metadata_i32_per_part),
        DType::I32,
        q.device(),
    )?;
    let num_splits = Tensor::zeros((meta.num_splits_len,), DType::I32, q.device())?;
    let lse_accum = Tensor::zeros(
        (total_num_splits, dims.h_k, q_seq_per_hk),
        DType::F32,
        q.device(),
    )?;
    let o_accum = Tensor::zeros(
        (total_num_splits, dims.h_k, q_seq_per_hk, dims.d_v),
        DType::F32,
        q.device(),
    )?;

    {
        let (seqlens_storage, seqlens_layout) = seqlens_k.storage_and_layout();
        let (scheduler_metadata_storage, scheduler_metadata_layout) =
            scheduler_metadata.storage_and_layout();
        let (num_splits_storage, num_splits_layout) = num_splits.storage_and_layout();

        let seqlens_ptr = tensor_ptr_i32(
            &seqlens_storage,
            seqlens_layout.start_offset(),
            &stream,
            "seqlens_k",
        )?;
        let scheduler_metadata_ptr = tensor_ptr_i32(
            &scheduler_metadata_storage,
            scheduler_metadata_layout.start_offset(),
            &stream,
            "scheduler_metadata",
        )?;
        let num_splits_ptr = tensor_ptr_i32(
            &num_splits_storage,
            num_splits_layout.start_offset(),
            &stream,
            "num_splits",
        )?;

        let plan_params = DenseDecodePlanParams {
            dims,
            seqlens_k: seqlens_ptr.as_const_i32(),
            tile_scheduler_metadata: scheduler_metadata_ptr.as_mut_i32(),
            num_splits: num_splits_ptr.as_mut_i32(),
            num_sm,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        let generated_meta = unsafe { flashmla_dense_decode_plan(&plan_params)? };
        if generated_meta != meta {
            return invalid_arg("dense decode metadata generation returned inconsistent plan");
        }
    }

    Ok(DenseDecodePlan {
        scheduler_metadata,
        num_splits,
        lse_accum,
        o_accum,
        meta,
    })
}

/// Launches FlashMLA dense decode on Candle CUDA tensors.
pub fn dense_decode(
    q: &Tensor,
    k_cache: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    plan: &mut DenseDecodePlan,
    config: DenseDecodeConfig,
) -> Result<DenseDecodeOutput> {
    let dims = validate_decode_tensors(q, k_cache, seqlens_k, Some(block_table), config)?;
    validate_plan_tensors(q, dims, plan)?;

    let q_heads_per_hk = dims.q_heads_per_hk()?;
    let q_seq_per_hk = dims.q_seq_per_hk()?;
    let q_packed = q
        .reshape((dims.batch, dims.s_q, dims.h_k, q_heads_per_hk, dims.d_qk))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((dims.batch, q_seq_per_hk, dims.h_k, dims.d_qk))?;
    let out_internal = Tensor::zeros(
        (dims.batch, dims.h_k, q_seq_per_hk, dims.d_v),
        DType::BF16,
        q.device(),
    )?;
    let lse_internal = Tensor::zeros((dims.batch, dims.h_k, q_seq_per_hk), DType::F32, q.device())?;

    let (stream, _device_id) = stream_and_device_id(q)?;
    let q_stride = q_packed.stride();
    let k_stride = k_cache.stride();
    let block_table_stride = block_table.stride();
    let strides = DenseDecodeStrides {
        q_b: q_stride[0],
        q_row: q_stride[1],
        q_head: q_stride[2],
        k_block: k_stride[0],
        k_row: k_stride[1],
        k_head: k_stride[2],
        block_table_b: block_table_stride[0],
    };

    {
        let (q_storage, q_layout) = q_packed.storage_and_layout();
        let (k_storage, k_layout) = k_cache.storage_and_layout();
        let (seqlens_storage, seqlens_layout) = seqlens_k.storage_and_layout();
        let (block_table_storage, block_table_layout) = block_table.storage_and_layout();
        let (out_storage, out_layout) = out_internal.storage_and_layout();
        let (lse_storage, lse_layout) = lse_internal.storage_and_layout();
        let (lse_accum_storage, lse_accum_layout) = plan.lse_accum.storage_and_layout();
        let (o_accum_storage, o_accum_layout) = plan.o_accum.storage_and_layout();
        let (scheduler_metadata_storage, scheduler_metadata_layout) =
            plan.scheduler_metadata.storage_and_layout();
        let (num_splits_storage, num_splits_layout) = plan.num_splits.storage_and_layout();

        let q_ptr = tensor_ptr_bf16(&q_storage, q_layout.start_offset(), &stream, "q_packed")?;
        let k_ptr = tensor_ptr_bf16(&k_storage, k_layout.start_offset(), &stream, "k_cache")?;
        let seqlens_ptr = tensor_ptr_i32(
            &seqlens_storage,
            seqlens_layout.start_offset(),
            &stream,
            "seqlens_k",
        )?;
        let block_table_ptr = tensor_ptr_i32(
            &block_table_storage,
            block_table_layout.start_offset(),
            &stream,
            "block_table",
        )?;
        let out_ptr = tensor_ptr_bf16(
            &out_storage,
            out_layout.start_offset(),
            &stream,
            "out_internal",
        )?;
        let lse_ptr = tensor_ptr_f32(
            &lse_storage,
            lse_layout.start_offset(),
            &stream,
            "lse_internal",
        )?;
        let lse_accum_ptr = tensor_ptr_f32(
            &lse_accum_storage,
            lse_accum_layout.start_offset(),
            &stream,
            "lse_accum",
        )?;
        let o_accum_ptr = tensor_ptr_f32(
            &o_accum_storage,
            o_accum_layout.start_offset(),
            &stream,
            "o_accum",
        )?;
        let scheduler_metadata_ptr = tensor_ptr_i32(
            &scheduler_metadata_storage,
            scheduler_metadata_layout.start_offset(),
            &stream,
            "scheduler_metadata",
        )?;
        let num_splits_ptr = tensor_ptr_i32(
            &num_splits_storage,
            num_splits_layout.start_offset(),
            &stream,
            "num_splits",
        )?;

        let params = DenseDecodeLaunchParams {
            dims,
            config,
            q: q_ptr.as_const_void(),
            kcache: k_ptr.as_const_void(),
            seqlens_k: seqlens_ptr.as_const_i32(),
            block_table: block_table_ptr.as_const_i32(),
            out: out_ptr.as_mut_void(),
            lse: lse_ptr.as_mut_f32(),
            lse_accum: lse_accum_ptr.as_mut_f32(),
            o_accum: o_accum_ptr.as_mut_f32(),
            strides,
            tile_scheduler_metadata: scheduler_metadata_ptr.as_mut_i32(),
            num_splits: num_splits_ptr.as_mut_i32(),
            num_sm_parts: plan.meta.num_sm_parts,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        unsafe { dense_decode_bf16(&params)? };
    }

    let out = out_internal
        .reshape((dims.batch, dims.h_k, dims.s_q, q_heads_per_hk, dims.d_v))?
        .transpose(1, 2)?
        .reshape((dims.batch, dims.s_q, dims.h_q, dims.d_v))?;
    let lse = lse_internal
        .reshape((dims.batch, dims.h_k, dims.s_q, q_heads_per_hk))?
        .transpose(2, 3)?
        .reshape((dims.batch, dims.h_q, dims.s_q))?;

    Ok(DenseDecodeOutput { out, lse })
}

fn validate_decode_tensors(
    q: &Tensor,
    k_cache: &Tensor,
    seqlens_k: &Tensor,
    block_table: Option<&Tensor>,
    config: DenseDecodeConfig,
) -> Result<DenseDecodeDims> {
    ensure_rank(q, 4, "q")?;
    ensure_rank(k_cache, 4, "k_cache")?;
    ensure_rank(seqlens_k, 1, "seqlens_k")?;
    ensure_last_dim_contiguous(q, "q")?;
    ensure_last_dim_contiguous(k_cache, "k_cache")?;
    ensure_last_dim_contiguous(seqlens_k, "seqlens_k")?;
    ensure_same_device(q, k_cache, "k_cache")?;
    ensure_same_device(q, seqlens_k, "seqlens_k")?;
    ensure_dtype(q, DType::BF16, "q")?;
    ensure_dtype(k_cache, DType::BF16, "k_cache")?;
    ensure_dtype(seqlens_k, DType::I32, "seqlens_k")?;

    let (batch, s_q, h_q, d_qk) = q.dims4()?;
    let (num_blocks, page_block_size, h_k, k_d_qk) = k_cache.dims4()?;
    if k_d_qk != d_qk {
        return invalid_arg(format!(
            "k_cache d_qk ({k_d_qk}) must match q d_qk ({d_qk})"
        ));
    }
    let seqlens_len = seqlens_k.dims1()?;
    if seqlens_len != batch {
        return invalid_arg(format!(
            "seqlens_k must have shape [{batch}], got [{seqlens_len}]"
        ));
    }

    if let Some(block_table) = block_table {
        ensure_rank(block_table, 2, "block_table")?;
        ensure_last_dim_contiguous(block_table, "block_table")?;
        ensure_same_device(q, block_table, "block_table")?;
        ensure_dtype(block_table, DType::I32, "block_table")?;
        let (table_batch, max_blocks) = block_table.dims2()?;
        if table_batch != batch {
            return invalid_arg(format!(
                "block_table batch must be {batch}, got {table_batch}"
            ));
        }
        if max_blocks == 0 {
            return invalid_arg("block_table must have at least one block column");
        }
    }

    let dims = DenseDecodeDims {
        batch,
        s_q,
        h_q,
        h_k,
        d_qk,
        d_v: config.d_v,
        num_blocks,
        page_block_size,
    };
    dims.validate()?;
    Ok(dims)
}

fn validate_plan_tensors(q: &Tensor, dims: DenseDecodeDims, plan: &DenseDecodePlan) -> Result<()> {
    ensure_rank(&plan.scheduler_metadata, 2, "scheduler_metadata")?;
    ensure_rank(&plan.num_splits, 1, "num_splits")?;
    ensure_rank(&plan.lse_accum, 3, "lse_accum")?;
    ensure_rank(&plan.o_accum, 4, "o_accum")?;
    ensure_same_device(q, &plan.scheduler_metadata, "scheduler_metadata")?;
    ensure_same_device(q, &plan.num_splits, "num_splits")?;
    ensure_same_device(q, &plan.lse_accum, "lse_accum")?;
    ensure_same_device(q, &plan.o_accum, "o_accum")?;
    ensure_dtype(&plan.scheduler_metadata, DType::I32, "scheduler_metadata")?;
    ensure_dtype(&plan.num_splits, DType::I32, "num_splits")?;
    ensure_dtype(&plan.lse_accum, DType::F32, "lse_accum")?;
    ensure_dtype(&plan.o_accum, DType::F32, "o_accum")?;
    ensure_contiguous(&plan.scheduler_metadata, "scheduler_metadata")?;
    ensure_contiguous(&plan.num_splits, "num_splits")?;
    ensure_contiguous(&plan.lse_accum, "lse_accum")?;
    ensure_contiguous(&plan.o_accum, "o_accum")?;

    let scheduler_shape = plan.scheduler_metadata.dims2()?;
    let expected_scheduler_cols = plan.meta.scheduler_metadata_i32_len / plan.meta.num_sm_parts;
    if scheduler_shape != (plan.meta.num_sm_parts, expected_scheduler_cols) {
        return invalid_arg(format!(
            "scheduler_metadata shape must be [{}, {}], got {:?}",
            plan.meta.num_sm_parts, expected_scheduler_cols, scheduler_shape
        ));
    }
    let num_splits_len = plan.num_splits.dims1()?;
    if num_splits_len != plan.meta.num_splits_len {
        return invalid_arg(format!(
            "num_splits length must be {}, got {num_splits_len}",
            plan.meta.num_splits_len
        ));
    }

    let total_num_splits = plan.meta.total_num_splits(dims)?;
    let q_seq_per_hk = dims.q_seq_per_hk()?;
    if plan.lse_accum.dims3()? != (total_num_splits, dims.h_k, q_seq_per_hk) {
        return invalid_arg("lse_accum shape does not match dense decode plan metadata");
    }
    if plan.o_accum.dims4()? != (total_num_splits, dims.h_k, q_seq_per_hk, dims.d_v) {
        return invalid_arg("o_accum shape does not match dense decode plan metadata");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device, Tensor};

    use super::*;

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn dense_decode_sm90_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = Tensor::zeros((1, 1, 1, 576), DType::BF16, &device)?;
        let k_cache = Tensor::zeros((1, 64, 1, 576), DType::BF16, &device)?;
        let seqlens_k = Tensor::zeros((1,), DType::I32, &device)?;
        let block_table = Tensor::zeros((1, 1), DType::I32, &device)?;
        let config = DenseDecodeConfig {
            softmax_scale: 1.0,
            d_v: 512,
            is_causal: false,
        };

        let mut plan = dense_decode_plan(&q, &k_cache, &seqlens_k, config)?;
        let output = dense_decode(&q, &k_cache, &seqlens_k, &block_table, &mut plan, config)?;

        assert_eq!(output.out.dims4()?, (1, 1, 1, 512));
        assert_eq!(output.out.dtype(), DType::BF16);
        assert_eq!(output.lse.dims3()?, (1, 1, 1));
        device.synchronize()?;

        Ok(())
    }
}

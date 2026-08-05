//! Sparse decode API re-exports and Candle CUDA integration.

pub use flashmla::{SparseDecodeConfig, SparseDecodeDims, SparseDecodePlanMeta};

use candle::{DType, Tensor};
use flashmla::{
    Arch, SparseDecodeLaunchParams, SparseDecodePlanParams, SparseDecodeStrides, get_device_info,
    sparse_decode_bf16_fp8, sparse_decode_plan as flashmla_sparse_decode_plan,
};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_last_dim_contiguous, ensure_rank, ensure_same_device,
        stream_and_device_id, tensor_ptr_bf16, tensor_ptr_f32, tensor_ptr_i32, tensor_ptr_u8_or_f8,
    },
};

/// Caller-owned sparse decode scheduler and split-KV workspaces.
#[derive(Debug)]
pub struct SparseDecodePlan {
    /// I32 scheduler metadata shaped `[num_sm_parts, metadata_i32_per_part]`.
    pub scheduler_metadata: Tensor,
    /// I32 split offsets shaped `[batch + 1]`.
    pub num_splits: Tensor,
    /// F32 split-KV LSE accumulation workspace.
    pub lse_accum: Tensor,
    /// F32 split-KV output accumulation workspace.
    pub o_accum: Tensor,
    /// Workspace sizing metadata returned by FlashMLA planning.
    pub meta: SparseDecodePlanMeta,
}

/// Output tensors returned by sparse decode.
#[derive(Debug)]
pub struct SparseDecodeOutput {
    /// BF16 attention output shaped `[batch, s_q, h_q, d_v]`.
    pub out: Tensor,
    /// F32 log-sum-exp values shaped `[batch, h_q, s_q]` to mirror upstream FlashMLA.
    pub lse: Tensor,
}

/// Allocates sparse decode scheduler metadata and split-KV workspaces.
pub fn sparse_decode_plan(
    q: &Tensor,
    kv_cache: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    extra_kv_cache: Option<&Tensor>,
    extra_indices: Option<&Tensor>,
    extra_topk_length: Option<&Tensor>,
    config: SparseDecodeConfig,
) -> Result<SparseDecodePlan> {
    let dims = validate_decode_tensors(
        q,
        kv_cache,
        indices,
        topk_length,
        None,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        config,
    )?;
    let (stream, device_id) = stream_and_device_id(q)?;
    let device = get_device_info(device_id)?;
    validate_kv_block_strides_for_arch(dims, kv_cache, extra_kv_cache, device.arch)?;
    let num_sm = usize::try_from(device.num_sms)
        .map_err(|_| crate::Error::Tensor("num_sms overflow".to_string()))?;
    let expected_meta = dims.plan_for_arch(device.arch, num_sm)?;

    let meta = {
        let topk_length_storage_and_layout = topk_length.map(Tensor::storage_and_layout);
        let extra_topk_length_storage_and_layout =
            extra_topk_length.map(Tensor::storage_and_layout);
        let topk_length_ptr = match &topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "topk_length",
            )?),
            None => None,
        };
        let extra_topk_length_ptr = match &extra_topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "extra_topk_length",
            )?),
            None => None,
        };

        let plan_params = SparseDecodePlanParams {
            dims,
            topk_length: topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            extra_topk_length: extra_topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            tile_scheduler_metadata: std::ptr::null_mut(),
            num_splits: std::ptr::null_mut(),
            num_sm,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        unsafe { flashmla_sparse_decode_plan(&plan_params)? }
    };
    if meta != expected_meta {
        return invalid_arg(
            "sparse decode FFI planning disagrees with architecture-specific workspace sizing",
        );
    }

    if meta.num_sm_parts == 0 || meta.scheduler_metadata_i32_len % meta.num_sm_parts != 0 {
        return invalid_arg("invalid sparse decode scheduler metadata shape");
    }
    let metadata_i32_per_part = meta.scheduler_metadata_i32_len / meta.num_sm_parts;
    let (lse_accum_shape, o_accum_shape) = decode_workspace_shapes(meta, dims)?;

    // SAFETY: The second planning call writes all scheduler metadata and split offsets. Decode
    // writes every accumulator element that combine can read; scalar no-split accumulators are
    // required to be non-null but are never accessed.
    let (scheduler_metadata, num_splits, lse_accum, o_accum) = unsafe {
        (
            Tensor::empty(
                (meta.num_sm_parts, metadata_i32_per_part),
                DType::I32,
                q.device(),
            )?,
            Tensor::empty((meta.num_splits_len,), DType::I32, q.device())?,
            Tensor::empty(lse_accum_shape, DType::F32, q.device())?,
            Tensor::empty(o_accum_shape, DType::F32, q.device())?,
        )
    };

    {
        let topk_length_storage_and_layout = topk_length.map(Tensor::storage_and_layout);
        let extra_topk_length_storage_and_layout =
            extra_topk_length.map(Tensor::storage_and_layout);
        let topk_length_ptr = match &topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "topk_length",
            )?),
            None => None,
        };
        let extra_topk_length_ptr = match &extra_topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "extra_topk_length",
            )?),
            None => None,
        };
        let (scheduler_metadata_storage, scheduler_metadata_layout) =
            scheduler_metadata.storage_and_layout();
        let (num_splits_storage, num_splits_layout) = num_splits.storage_and_layout();
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
        let plan_params = SparseDecodePlanParams {
            dims,
            topk_length: topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            extra_topk_length: extra_topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            tile_scheduler_metadata: scheduler_metadata_ptr.as_mut_i32(),
            num_splits: num_splits_ptr.as_mut_i32(),
            num_sm,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };
        let generated_meta = unsafe { flashmla_sparse_decode_plan(&plan_params)? };
        if generated_meta != meta {
            return invalid_arg(
                "sparse decode metadata generation returned inconsistent plan metadata",
            );
        }
    }

    Ok(SparseDecodePlan {
        scheduler_metadata,
        num_splits,
        lse_accum,
        o_accum,
        meta,
    })
}

/// Launches FlashMLA sparse decode on Candle CUDA tensors.
pub fn sparse_decode(
    q: &Tensor,
    kv_cache: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    attn_sink: Option<&Tensor>,
    extra_kv_cache: Option<&Tensor>,
    extra_indices: Option<&Tensor>,
    extra_topk_length: Option<&Tensor>,
    plan: &mut SparseDecodePlan,
    config: SparseDecodeConfig,
) -> Result<SparseDecodeOutput> {
    let dims = validate_decode_tensors(
        q,
        kv_cache,
        indices,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        config,
    )?;
    validate_plan_tensors(q, dims, plan)?;

    // SAFETY: Sparse decode writes every output and LSE element before returning these tensors.
    let (out, lse_internal) = unsafe {
        (
            Tensor::empty(
                (dims.batch, dims.s_q, dims.h_q, dims.d_v),
                DType::BF16,
                q.device(),
            )?,
            Tensor::empty((dims.batch, dims.s_q, dims.h_q), DType::F32, q.device())?,
        )
    };

    let (stream, _device_id) = stream_and_device_id(q)?;
    let q_stride = q.stride();
    let kv_stride = kv_cache.stride();
    let indices_stride = indices.stride();
    let lse_stride = lse_internal.stride();
    let out_stride = out.stride();
    let lse_accum_stride = plan.lse_accum.stride();
    let o_accum_stride = plan.o_accum.stride();
    let (extra_kv_block, extra_kv_row, extra_indices_b, extra_indices_s_q) =
        match (extra_kv_cache, extra_indices) {
            (Some(extra_kv_cache), Some(extra_indices)) => {
                let extra_kv_stride = extra_kv_cache.stride();
                let extra_indices_stride = extra_indices.stride();
                (
                    extra_kv_stride[0],
                    extra_kv_stride[1],
                    extra_indices_stride[0],
                    extra_indices_stride[1],
                )
            }
            _ => (0, 0, 0, 0),
        };

    let strides = SparseDecodeStrides {
        q_b: q_stride[0],
        q_s_q: q_stride[1],
        q_h_q: q_stride[2],
        kv_block: kv_stride[0],
        kv_row: kv_stride[1],
        indices_b: indices_stride[0],
        indices_s_q: indices_stride[1],
        lse_b: lse_stride[0],
        lse_s_q: lse_stride[1],
        out_b: out_stride[0],
        out_s_q: out_stride[1],
        out_h_q: out_stride[2],
        extra_kv_block,
        extra_kv_row,
        extra_indices_b,
        extra_indices_s_q,
        lse_accum_split: lse_accum_stride[0],
        lse_accum_s_q: lse_accum_stride[1],
        o_accum_split: o_accum_stride[0],
        o_accum_s_q: o_accum_stride[1],
        o_accum_h_q: o_accum_stride[2],
    };

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (kv_cache_storage, kv_cache_layout) = kv_cache.storage_and_layout();
        let (indices_storage, indices_layout) = indices.storage_and_layout();
        let topk_length_storage_and_layout = topk_length.map(Tensor::storage_and_layout);
        let attn_sink_storage_and_layout = attn_sink.map(Tensor::storage_and_layout);
        let (out_storage, out_layout) = out.storage_and_layout();
        let (lse_storage, lse_layout) = lse_internal.storage_and_layout();
        let extra_kv_cache_storage_and_layout = extra_kv_cache.map(Tensor::storage_and_layout);
        let extra_indices_storage_and_layout = extra_indices.map(Tensor::storage_and_layout);
        let extra_topk_length_storage_and_layout =
            extra_topk_length.map(Tensor::storage_and_layout);
        let (lse_accum_storage, lse_accum_layout) = plan.lse_accum.storage_and_layout();
        let (o_accum_storage, o_accum_layout) = plan.o_accum.storage_and_layout();
        let (scheduler_metadata_storage, scheduler_metadata_layout) =
            plan.scheduler_metadata.storage_and_layout();
        let (num_splits_storage, num_splits_layout) = plan.num_splits.storage_and_layout();

        let q_ptr = tensor_ptr_bf16(&q_storage, q_layout.start_offset(), &stream, "q")?;
        let kv_ptr = tensor_ptr_u8_or_f8(
            &kv_cache_storage,
            kv_cache.dtype(),
            kv_cache_layout.start_offset(),
            &stream,
            "kv_cache",
        )?;
        let indices_ptr = tensor_ptr_i32(
            &indices_storage,
            indices_layout.start_offset(),
            &stream,
            "indices",
        )?;
        let topk_length_ptr = match &topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "topk_length",
            )?),
            None => None,
        };
        let attn_sink_ptr = match &attn_sink_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_f32(
                storage,
                layout.start_offset(),
                &stream,
                "attn_sink",
            )?),
            None => None,
        };
        let out_ptr = tensor_ptr_bf16(&out_storage, out_layout.start_offset(), &stream, "out")?;
        let lse_ptr = tensor_ptr_f32(&lse_storage, lse_layout.start_offset(), &stream, "lse")?;
        let extra_kv_ptr = match &extra_kv_cache_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_u8_or_f8(
                storage,
                extra_kv_cache
                    .expect("extra_kv_cache storage exists only when tensor exists")
                    .dtype(),
                layout.start_offset(),
                &stream,
                "extra_kv_cache",
            )?),
            None => None,
        };
        let extra_indices_ptr = match &extra_indices_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "extra_indices",
            )?),
            None => None,
        };
        let extra_topk_length_ptr = match &extra_topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "extra_topk_length",
            )?),
            None => None,
        };
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

        let params = SparseDecodeLaunchParams {
            dims,
            config,
            q: q_ptr.as_const_void(),
            kv: kv_ptr.as_const_void(),
            indices: indices_ptr.as_const_i32(),
            topk_length: topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            attn_sink: attn_sink_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_f32()),
            out: out_ptr.as_mut_void(),
            lse: lse_ptr.as_mut_f32(),
            extra_kv: extra_kv_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_void()),
            extra_indices: extra_indices_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            extra_topk_length: extra_topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            strides,
            lse_accum: lse_accum_ptr.as_mut_f32(),
            o_accum: o_accum_ptr.as_mut_f32(),
            tile_scheduler_metadata: scheduler_metadata_ptr.as_mut_i32(),
            num_splits: num_splits_ptr.as_mut_i32(),
            num_sm_parts: plan.meta.num_sm_parts,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        unsafe { sparse_decode_bf16_fp8(&params)? };
    }

    Ok(SparseDecodeOutput {
        out,
        lse: lse_internal.transpose(1, 2)?,
    })
}

fn validate_decode_tensors(
    q: &Tensor,
    kv_cache: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    attn_sink: Option<&Tensor>,
    extra_kv_cache: Option<&Tensor>,
    extra_indices: Option<&Tensor>,
    extra_topk_length: Option<&Tensor>,
    config: SparseDecodeConfig,
) -> Result<SparseDecodeDims> {
    ensure_rank(q, 4, "q")?;
    ensure_rank(kv_cache, 4, "kv_cache")?;
    ensure_rank(indices, 3, "indices")?;
    ensure_last_dim_contiguous(q, "q")?;
    ensure_last_dim_contiguous(kv_cache, "kv_cache")?;
    ensure_last_dim_contiguous(indices, "indices")?;
    ensure_same_device(q, kv_cache, "kv_cache")?;
    ensure_same_device(q, indices, "indices")?;
    ensure_dtype(q, DType::BF16, "q")?;
    ensure_kv_cache_dtype(kv_cache, "kv_cache")?;
    ensure_dtype(indices, DType::I32, "indices")?;

    let (batch, s_q, h_q, d_qk) = q.dims4()?;
    let (num_blocks, page_block_size, h_kv, bytes_per_token) = kv_cache.dims4()?;
    let (indices_batch, indices_s_q, topk) = indices.dims3()?;
    if indices_batch != batch || indices_s_q != s_q {
        return invalid_arg(format!(
            "indices must have shape [{batch}, {s_q}, topk], got [{indices_batch}, {indices_s_q}, {topk}]"
        ));
    }

    if let Some(topk_length) = topk_length {
        ensure_rank(topk_length, 1, "topk_length")?;
        ensure_last_dim_contiguous(topk_length, "topk_length")?;
        ensure_same_device(q, topk_length, "topk_length")?;
        ensure_dtype(topk_length, DType::I32, "topk_length")?;
        let len = topk_length.dims1()?;
        if len != batch {
            return invalid_arg(format!(
                "topk_length must have shape [{batch}], got [{len}]"
            ));
        }
    }

    if let Some(attn_sink) = attn_sink {
        ensure_rank(attn_sink, 1, "attn_sink")?;
        ensure_last_dim_contiguous(attn_sink, "attn_sink")?;
        ensure_same_device(q, attn_sink, "attn_sink")?;
        ensure_dtype(attn_sink, DType::F32, "attn_sink")?;
        let len = attn_sink.dims1()?;
        if len != h_q {
            return invalid_arg(format!("attn_sink must have shape [{h_q}], got [{len}]"));
        }
    }

    let (extra_num_blocks, extra_page_block_size, extra_topk) = validate_extra_tensors(
        q,
        kv_cache,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    )?;

    let dims = SparseDecodeDims {
        batch,
        s_q,
        h_q,
        h_kv,
        d_qk,
        d_v: config.d_v,
        num_blocks,
        page_block_size,
        topk,
        extra_num_blocks,
        extra_page_block_size,
        extra_topk,
    };
    dims.validate()?;

    let expected_bytes_per_token = dims.kv_bytes_per_token()?;
    if bytes_per_token != expected_bytes_per_token {
        return invalid_arg(format!(
            "kv_cache last dim must be {expected_bytes_per_token} bytes for d_qk={d_qk}, got {bytes_per_token}"
        ));
    }
    if kv_cache.stride()[1] != expected_bytes_per_token {
        return invalid_arg(format!(
            "kv_cache page rows must be contiguous with stride {expected_bytes_per_token}, got {}",
            kv_cache.stride()[1]
        ));
    }
    if d_qk == 576 && topk_length.is_some() {
        return invalid_arg("V32 sparse decode does not support topk_length");
    }

    Ok(dims)
}

fn validate_kv_block_strides_for_arch(
    dims: SparseDecodeDims,
    kv_cache: &Tensor,
    extra_kv_cache: Option<&Tensor>,
    arch: Arch,
) -> Result<()> {
    if arch != Arch::Sm100f {
        return Ok(());
    }
    let required = if dims.d_qk == 576 { 656 } else { 576 };
    if kv_cache.stride()[0] % required != 0 {
        return invalid_arg(format!(
            "SM100 kv_cache block stride must be a multiple of {required}, got {}; pad each cache block as needed",
            kv_cache.stride()[0]
        ));
    }
    if let Some(extra_kv_cache) = extra_kv_cache
        && extra_kv_cache.stride()[0] % required != 0
    {
        return invalid_arg(format!(
            "SM100 extra_kv_cache block stride must be a multiple of {required}, got {}; pad each cache block as needed",
            extra_kv_cache.stride()[0]
        ));
    }
    Ok(())
}

fn validate_extra_tensors(
    q: &Tensor,
    kv_cache: &Tensor,
    extra_kv_cache: Option<&Tensor>,
    extra_indices: Option<&Tensor>,
    extra_topk_length: Option<&Tensor>,
) -> Result<(usize, usize, usize)> {
    match (extra_kv_cache, extra_indices) {
        (None, None) => {
            if extra_topk_length.is_some() {
                return invalid_arg("extra_topk_length requires extra_kv_cache and extra_indices");
            }
            Ok((0, 0, 0))
        }
        (Some(extra_kv_cache), Some(extra_indices)) => {
            ensure_rank(extra_kv_cache, 4, "extra_kv_cache")?;
            ensure_rank(extra_indices, 3, "extra_indices")?;
            ensure_last_dim_contiguous(extra_kv_cache, "extra_kv_cache")?;
            ensure_last_dim_contiguous(extra_indices, "extra_indices")?;
            ensure_same_device(q, extra_kv_cache, "extra_kv_cache")?;
            ensure_same_device(q, extra_indices, "extra_indices")?;
            ensure_kv_cache_dtype(extra_kv_cache, "extra_kv_cache")?;
            ensure_dtype(extra_indices, DType::I32, "extra_indices")?;
            if extra_kv_cache.dtype() != kv_cache.dtype() {
                return invalid_arg(format!(
                    "extra_kv_cache dtype must match kv_cache dtype {:?}, got {:?}",
                    kv_cache.dtype(),
                    extra_kv_cache.dtype()
                ));
            }

            let (batch, s_q, _h_q, d_qk) = q.dims4()?;
            let (_num_blocks, _page_block_size, h_kv, bytes_per_token) = kv_cache.dims4()?;
            let (extra_num_blocks, extra_page_block_size, extra_h_kv, extra_bytes_per_token) =
                extra_kv_cache.dims4()?;
            if extra_h_kv != h_kv || extra_bytes_per_token != bytes_per_token {
                return invalid_arg(format!(
                    "extra_kv_cache must match h_kv={h_kv} and bytes_per_token={bytes_per_token}, got h_kv={extra_h_kv} bytes={extra_bytes_per_token}"
                ));
            }
            if extra_kv_cache.stride()[1] != bytes_per_token {
                return invalid_arg(format!(
                    "extra_kv_cache page rows must be contiguous with stride {bytes_per_token}, got {}",
                    extra_kv_cache.stride()[1]
                ));
            }

            let (extra_indices_batch, extra_indices_s_q, extra_topk) = extra_indices.dims3()?;
            if extra_indices_batch != batch || extra_indices_s_q != s_q {
                return invalid_arg(format!(
                    "extra_indices must have shape [{batch}, {s_q}, extra_topk], got [{extra_indices_batch}, {extra_indices_s_q}, {extra_topk}]"
                ));
            }
            if d_qk == 576 {
                return invalid_arg("V32 sparse decode does not support extra KV cache");
            }

            if let Some(extra_topk_length) = extra_topk_length {
                ensure_rank(extra_topk_length, 1, "extra_topk_length")?;
                ensure_last_dim_contiguous(extra_topk_length, "extra_topk_length")?;
                ensure_same_device(q, extra_topk_length, "extra_topk_length")?;
                ensure_dtype(extra_topk_length, DType::I32, "extra_topk_length")?;
                let len = extra_topk_length.dims1()?;
                if len != batch {
                    return invalid_arg(format!(
                        "extra_topk_length must have shape [{batch}], got [{len}]"
                    ));
                }
            }

            Ok((extra_num_blocks, extra_page_block_size, extra_topk))
        }
        _ => invalid_arg("extra_kv_cache and extra_indices must be provided together"),
    }
}

fn validate_plan_tensors(
    q: &Tensor,
    dims: SparseDecodeDims,
    plan: &SparseDecodePlan,
) -> Result<()> {
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
    ensure_last_dim_contiguous(&plan.scheduler_metadata, "scheduler_metadata")?;
    ensure_last_dim_contiguous(&plan.num_splits, "num_splits")?;
    ensure_last_dim_contiguous(&plan.lse_accum, "lse_accum")?;
    ensure_last_dim_contiguous(&plan.o_accum, "o_accum")?;

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

    let (lse_accum_shape, o_accum_shape) = decode_workspace_shapes(plan.meta, dims)?;
    if plan.lse_accum.dims3()? != lse_accum_shape {
        return invalid_arg("lse_accum shape does not match sparse decode plan metadata");
    }
    if plan.o_accum.dims4()? != o_accum_shape {
        return invalid_arg("o_accum shape does not match sparse decode plan metadata");
    }
    Ok(())
}

fn decode_workspace_shapes(
    meta: SparseDecodePlanMeta,
    dims: SparseDecodeDims,
) -> Result<((usize, usize, usize), (usize, usize, usize, usize))> {
    if meta.num_sm_parts == 1 {
        if meta.lse_accum_elem_count != 1 || meta.o_accum_elem_count != 1 {
            return invalid_arg("no-split sparse decode plan must use scalar accumulators");
        }
        return Ok(((1, 1, 1), (1, 1, 1, 1)));
    }

    let total_num_splits = meta.total_num_splits(dims)?;
    let lse_shape = (total_num_splits, dims.s_q, dims.h_q);
    let o_shape = (total_num_splits, dims.s_q, dims.h_q, dims.d_v);
    let lse_elems = total_num_splits
        .checked_mul(dims.s_q)
        .and_then(|value| value.checked_mul(dims.h_q))
        .ok_or_else(|| crate::Error::Tensor("sparse decode LSE workspace overflow".to_string()))?;
    let o_elems = lse_elems.checked_mul(dims.d_v).ok_or_else(|| {
        crate::Error::Tensor("sparse decode output workspace overflow".to_string())
    })?;
    if meta.lse_accum_elem_count != lse_elems || meta.o_accum_elem_count != o_elems {
        return invalid_arg("sparse decode accumulator sizing metadata is inconsistent");
    }
    Ok((lse_shape, o_shape))
}

fn ensure_kv_cache_dtype(t: &Tensor, name: &str) -> Result<()> {
    match t.dtype() {
        DType::U8 | DType::F8E4M3 => Ok(()),
        dtype => invalid_arg(format!(
            "{name} must have dtype U8 or F8E4M3, got {dtype:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device, Tensor};
    use half::bf16;

    use super::*;

    fn test_decode_dims() -> SparseDecodeDims {
        SparseDecodeDims {
            batch: 1,
            s_q: 1024,
            h_q: 64,
            h_kv: 1,
            d_qk: 576,
            d_v: 512,
            num_blocks: 1,
            page_block_size: 64,
            topk: 2048,
            extra_num_blocks: 0,
            extra_page_block_size: 0,
            extra_topk: 0,
        }
    }

    fn test_plan_meta(
        num_sm_parts: usize,
        lse_accum_elem_count: usize,
        o_accum_elem_count: usize,
    ) -> SparseDecodePlanMeta {
        SparseDecodePlanMeta {
            num_sm_parts,
            fixed_overhead_num_blocks: 5,
            block_size_topk: 64,
            scheduler_metadata_i32_len: num_sm_parts * 8,
            num_splits_len: 2,
            lse_accum_elem_count,
            o_accum_elem_count,
        }
    }

    fn v32_unit_value_cache(page_block_size: usize) -> Vec<u8> {
        const BYTES_PER_TOKEN: usize = 656;
        let mut cache = vec![0_u8; page_block_size * BYTES_PER_TOKEN];
        for row in cache.chunks_exact_mut(BYTES_PER_TOKEN) {
            row[..512].fill(0x38); // FP8 E4M3 encoding of 1.0.
            for scale in row[512..528].chunks_exact_mut(4) {
                scale.copy_from_slice(&1.0_f32.to_ne_bytes());
            }
            // The final 128 bytes are the 64 BF16 RoPE dimensions and remain zero.
        }
        cache
    }

    fn model1_unit_value_cache(page_block_size: usize) -> Vec<u8> {
        const MAIN_BYTES_PER_TOKEN: usize = 576;
        const BYTES_PER_TOKEN: usize = 584;
        let mut cache = vec![0_u8; page_block_size * BYTES_PER_TOKEN];
        for token in 0..page_block_size {
            let main_start = token * MAIN_BYTES_PER_TOKEN;
            cache[main_start..main_start + 448].fill(0x38); // FP8 E4M3 1.0.
        }
        let scales_start = page_block_size * MAIN_BYTES_PER_TOKEN;
        for token in 0..page_block_size {
            let token_scales = scales_start + token * 8;
            cache[token_scales..token_scales + 7].fill(0x7f); // FP8 E8M0 1.0.
        }
        cache
    }

    fn assert_unit_decode_output(output: &SparseDecodeOutput) -> Result<()> {
        let values = output.out.flatten_all()?.to_vec1::<bf16>()?;
        assert!(values.iter().all(|value| f32::from(*value).is_finite()));
        assert!(
            values
                .iter()
                .all(|value| (f32::from(*value) - 1.0).abs() <= 0.03)
        );
        let lse = output.lse.flatten_all()?.to_vec1::<f32>()?;
        let expected_lse = (64.0_f32).ln();
        assert!(lse.iter().all(|value| value.is_finite()));
        assert!(lse.iter().all(|value| (value - expected_lse).abs() <= 0.03));
        Ok(())
    }

    fn assert_model1_decode_output(output: &SparseDecodeOutput) -> Result<()> {
        let values = output.out.flatten_all()?.to_vec1::<bf16>()?;
        for value_head in values.chunks_exact(512) {
            assert!(
                value_head[..448]
                    .iter()
                    .all(|value| (f32::from(*value) - 1.0).abs() <= 0.03)
            );
            assert!(
                value_head[448..]
                    .iter()
                    .all(|value| f32::from(*value).abs() <= 0.03)
            );
        }
        let lse = output.lse.flatten_all()?.to_vec1::<f32>()?;
        let expected_lse = (64.0_f32).ln();
        assert!(
            lse.iter()
                .all(|value| value.is_finite() && (value - expected_lse).abs() <= 0.03)
        );
        Ok(())
    }

    #[test]
    fn no_split_decode_uses_scalar_workspaces() -> Result<()> {
        let shapes = decode_workspace_shapes(test_plan_meta(1, 1, 1), test_decode_dims())?;
        assert_eq!(shapes, ((1, 1, 1), (1, 1, 1, 1)));
        Ok(())
    }

    #[test]
    fn split_decode_keeps_full_workspaces() -> Result<()> {
        let dims = test_decode_dims();
        let total_splits = dims.batch + 2;
        let lse_elems = total_splits * dims.s_q * dims.h_q;
        let o_elems = lse_elems * dims.d_v;
        let shapes = decode_workspace_shapes(test_plan_meta(2, lse_elems, o_elems), dims)?;
        assert_eq!(
            shapes,
            (
                (total_splits, dims.s_q, dims.h_q),
                (total_splits, dims.s_q, dims.h_q, dims.d_v)
            )
        );
        Ok(())
    }

    #[test]
    fn sm100_model1_requires_tma_aligned_cache_blocks() -> Result<()> {
        let dims = SparseDecodeDims {
            h_q: 128,
            d_qk: 512,
            ..test_decode_dims()
        };
        let contiguous = Tensor::zeros((1, 64, 1, 584), DType::U8, &Device::Cpu)?;
        assert!(validate_kv_block_strides_for_arch(dims, &contiguous, None, Arch::Sm100f).is_err());

        let padded = Tensor::zeros((1, 72, 1, 584), DType::U8, &Device::Cpu)?.narrow(1, 0, 64)?;
        validate_kv_block_strides_for_arch(dims, &padded, None, Arch::Sm100f)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn sparse_decode_sm90_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = Tensor::zeros((1, 1, 64, 576), DType::BF16, &device)?;
        let kv_cache = Tensor::zeros((1, 64, 1, 656), DType::U8, &device)?;
        let indices = Tensor::zeros((1, 1, 64), DType::I32, &device)?;
        let mut plan = sparse_decode_plan(
            &q,
            &kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            SparseDecodeConfig {
                softmax_scale: 1.0,
                d_v: 512,
                pad_heads: false,
            },
        )?;
        let output = sparse_decode(
            &q,
            &kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            None,
            &mut plan,
            SparseDecodeConfig {
                softmax_scale: 1.0,
                d_v: 512,
                pad_heads: false,
            },
        )?;

        assert_eq!(output.out.dims4()?, (1, 1, 64, 512));
        assert_eq!(output.out.dtype(), DType::BF16);
        assert_eq!(output.lse.dims3()?, (1, 64, 1));
        device.synchronize()?;

        Ok(())
    }

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
    fn sparse_decode_sm90_no_split_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = Tensor::zeros((1, 1024, 64, 576), DType::BF16, &device)?;
        let kv_cache = Tensor::zeros((1, 64, 1, 656), DType::U8, &device)?;
        let indices = Tensor::zeros((1, 1024, 64), DType::I32, &device)?;
        let mut plan = sparse_decode_plan(
            &q,
            &kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            SparseDecodeConfig {
                softmax_scale: 1.0,
                d_v: 512,
                pad_heads: false,
            },
        )?;
        assert_eq!(plan.meta.num_sm_parts, 1);
        assert_eq!(plan.meta.lse_accum_elem_count, 1);
        assert_eq!(plan.meta.o_accum_elem_count, 1);
        assert_eq!(plan.lse_accum.dims3()?, (1, 1, 1));
        assert_eq!(plan.o_accum.dims4()?, (1, 1, 1, 1));

        let output = sparse_decode(
            &q,
            &kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            None,
            &mut plan,
            SparseDecodeConfig {
                softmax_scale: 1.0,
                d_v: 512,
                pad_heads: false,
            },
        )?;
        assert_eq!(output.out.dims4()?, (1, 1024, 64, 512));
        assert_eq!(output.lse.dims3()?, (1, 64, 1024));
        device.synchronize()?;

        Ok(())
    }

    #[test]
    #[ignore = "requires a visible SM100 CUDA GPU"]
    fn sparse_decode_sm100_smoke() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags, CUstreamCaptureMode,
        };

        let device = Device::new_cuda_with_stream(0)?;
        let cuda_device = device.as_cuda_device()?;
        // SAFETY: This smoke test uses one non-default stream and manages ordering explicitly,
        // matching modeld's CUDA graph setup.
        unsafe { cuda_device.disable_event_tracking() };
        let info = get_device_info(0)?;
        assert_eq!(info.arch, flashmla::Arch::Sm100f);

        let q = Tensor::from_vec(vec![bf16::ZERO; 128 * 576], (1, 1, 128, 576), &device)?;
        let kv_cache = Tensor::from_vec(v32_unit_value_cache(64), (1, 64, 1, 656), &device)?;
        let indices = Tensor::from_vec((0_i32..64_i32).collect(), (1, 1, 64), &device)?;
        let config = SparseDecodeConfig {
            softmax_scale: 1.0,
            d_v: 512,
            pad_heads: false,
        };
        let mut plan = sparse_decode_plan(&q, &kv_cache, &indices, None, None, None, None, config)?;
        assert_eq!(plan.meta.fixed_overhead_num_blocks, 5);

        let warmup = sparse_decode(
            &q, &kv_cache, &indices, None, None, None, None, None, &mut plan, config,
        )?;
        device.synchronize()?;
        assert_unit_decode_output(&warmup)?;

        // Modeld captures intermediate allocations, copies its final output into stable memory,
        // and records frees before graph instantiation. Mirror that lifecycle here so replay is
        // tested rather than retaining a graph allocation outside the capture region.
        let stable_out = unsafe { Tensor::empty((1, 1, 128, 512), DType::BF16, &device)? };
        let stable_lse = unsafe { Tensor::empty((1, 128, 1), DType::F32, &device)? };
        let stream = cuda_device.cuda_stream();
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|error| {
                crate::Error::Tensor(format!("failed to begin CUDA graph capture: {error}"))
            })?;
        let captured_output = sparse_decode(
            &q, &kv_cache, &indices, None, None, None, None, None, &mut plan, config,
        )?;
        {
            use candle::cuda_backend::cudarc::driver::result::memcpy_dtod_async;

            let (captured_out_storage, captured_out_layout) =
                captured_output.out.storage_and_layout();
            let (stable_out_storage, stable_out_layout) = stable_out.storage_and_layout();
            let captured_out_ptr = tensor_ptr_bf16(
                &captured_out_storage,
                captured_out_layout.start_offset(),
                &stream,
                "captured_out",
            )?;
            let stable_out_ptr = tensor_ptr_bf16(
                &stable_out_storage,
                stable_out_layout.start_offset(),
                &stream,
                "stable_out",
            )?;

            let (captured_lse_storage, captured_lse_layout) =
                captured_output.lse.storage_and_layout();
            let (stable_lse_storage, stable_lse_layout) = stable_lse.storage_and_layout();
            let captured_lse_ptr = tensor_ptr_f32(
                &captured_lse_storage,
                captured_lse_layout.start_offset(),
                &stream,
                "captured_lse",
            )?;
            let stable_lse_ptr = tensor_ptr_f32(
                &stable_lse_storage,
                stable_lse_layout.start_offset(),
                &stream,
                "stable_lse",
            )?;

            // SAFETY: All four pointers are allocations on `stream`; destinations are large
            // enough for the exact captured output byte counts and remain live through replay.
            unsafe {
                memcpy_dtod_async(
                    stable_out_ptr.as_mut_void() as usize as u64,
                    captured_out_ptr.as_const_void() as usize as u64,
                    128 * 512 * std::mem::size_of::<bf16>(),
                    stream.cu_stream(),
                )
                .map_err(|error| {
                    crate::Error::Tensor(format!("failed to capture output copy: {error}"))
                })?;
                memcpy_dtod_async(
                    stable_lse_ptr.as_mut_f32() as usize as u64,
                    captured_lse_ptr.as_const_f32() as usize as u64,
                    128 * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
                .map_err(|error| {
                    crate::Error::Tensor(format!("failed to capture LSE copy: {error}"))
                })?;
            }
        }
        drop(captured_output);
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|error| {
                crate::Error::Tensor(format!("failed to end CUDA graph capture: {error}"))
            })?
            .ok_or_else(|| crate::Error::Tensor("CUDA graph capture was empty".to_string()))?;
        graph.launch().map_err(|error| {
            crate::Error::Tensor(format!("failed to replay CUDA graph: {error}"))
        })?;
        device.synchronize()?;
        let stable_output = SparseDecodeOutput {
            out: stable_out,
            lse: stable_lse,
        };
        assert_unit_decode_output(&stable_output)?;
        graph.launch().map_err(|error| {
            crate::Error::Tensor(format!(
                "failed to replay CUDA graph a second time: {error}"
            ))
        })?;
        device.synchronize()?;
        assert_unit_decode_output(&stable_output)?;

        let model1_q = Tensor::from_vec(vec![bf16::ZERO; 128 * 512], (1, 1, 128, 512), &device)?;
        let mut model1_cache_data = model1_unit_value_cache(64);
        model1_cache_data.resize(72 * 584, 0);
        let model1_kv_cache =
            Tensor::from_vec(model1_cache_data, (1, 72, 1, 584), &device)?.narrow(1, 0, 64)?;
        assert_eq!(model1_kv_cache.stride()[0] % 576, 0);
        let mut model1_plan = sparse_decode_plan(
            &model1_q,
            &model1_kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            config,
        )?;
        assert_eq!(model1_plan.meta.fixed_overhead_num_blocks, 3);
        let model1_output = sparse_decode(
            &model1_q,
            &model1_kv_cache,
            &indices,
            None,
            None,
            None,
            None,
            None,
            &mut model1_plan,
            config,
        )?;
        device.synchronize()?;
        assert_model1_decode_output(&model1_output)?;

        Ok(())
    }
}

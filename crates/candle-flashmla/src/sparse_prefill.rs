//! Sparse prefill API re-exports for Candle integration.

pub use flashmla::{SparsePrefillConfig, SparsePrefillDims, SparsePrefillStrides};

#[cfg(feature = "cuda")]
use candle::{DType, Tensor};
#[cfg(feature = "cuda")]
use flashmla::{SparsePrefillLaunchParams, get_device_info, sparse_prefill_bf16};

#[cfg(feature = "cuda")]
use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_last_dim_contiguous, ensure_rank, ensure_same_device,
        stream_and_device_id, tensor_mut_ptr_bf16, tensor_mut_ptr_f32, tensor_ptr_bf16,
        tensor_ptr_f32, tensor_ptr_i32,
    },
};

/// Output tensors returned by sparse prefill.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct SparsePrefillOutput {
    /// BF16 attention output shaped `[s_q, h_q, d_v]`.
    pub out: Tensor,
    /// F32 per-query/head max logits shaped `[s_q, h_q]`.
    pub max_logits: Tensor,
    /// F32 per-query/head log-sum-exp values shaped `[s_q, h_q]`.
    pub lse: Tensor,
}

/// Launches FlashMLA sparse prefill on Candle CUDA tensors.
#[cfg(feature = "cuda")]
pub fn sparse_prefill(
    q: &Tensor,
    kv: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    attn_sink: Option<&Tensor>,
    config: SparsePrefillConfig,
) -> Result<SparsePrefillOutput> {
    ensure_rank(q, 3, "q")?;
    ensure_rank(kv, 3, "kv")?;
    ensure_rank(indices, 3, "indices")?;
    ensure_last_dim_contiguous(q, "q")?;
    ensure_last_dim_contiguous(kv, "kv")?;
    ensure_last_dim_contiguous(indices, "indices")?;
    ensure_same_device(q, kv, "kv")?;
    ensure_same_device(q, indices, "indices")?;
    ensure_dtype(q, DType::BF16, "q")?;
    ensure_dtype(kv, DType::BF16, "kv")?;
    ensure_dtype(indices, DType::I32, "indices")?;

    if let Some(topk_length) = topk_length {
        ensure_rank(topk_length, 1, "topk_length")?;
        ensure_last_dim_contiguous(topk_length, "topk_length")?;
        ensure_same_device(q, topk_length, "topk_length")?;
        ensure_dtype(topk_length, DType::I32, "topk_length")?;
    }
    if let Some(attn_sink) = attn_sink {
        ensure_rank(attn_sink, 1, "attn_sink")?;
        ensure_last_dim_contiguous(attn_sink, "attn_sink")?;
        ensure_same_device(q, attn_sink, "attn_sink")?;
        ensure_dtype(attn_sink, DType::F32, "attn_sink")?;
    }

    let (s_q, h_q, d_qk) = q.dims3()?;
    let (s_kv, h_kv, kv_d_qk) = kv.dims3()?;
    let (indices_s_q, indices_h_kv, topk) = indices.dims3()?;
    if kv_d_qk != d_qk {
        return invalid_arg(format!("kv d_qk ({kv_d_qk}) must match q d_qk ({d_qk})"));
    }
    if indices_s_q != s_q || indices_h_kv != h_kv {
        return invalid_arg(format!(
            "indices shape must start with [{s_q}, {h_kv}], got [{indices_s_q}, {indices_h_kv}]"
        ));
    }
    if let Some(topk_length) = topk_length {
        let len = topk_length.dims1()?;
        if len != s_q {
            return invalid_arg(format!("topk_length must have shape [{s_q}], got [{len}]"));
        }
    }
    if let Some(attn_sink) = attn_sink {
        let len = attn_sink.dims1()?;
        if len != h_q {
            return invalid_arg(format!("attn_sink must have shape [{h_q}], got [{len}]"));
        }
    }

    let dims = SparsePrefillDims {
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v: config.d_v,
        topk,
    };
    dims.validate()?;

    let out = Tensor::zeros((s_q, h_q, config.d_v), DType::BF16, q.device())?;
    let max_logits = Tensor::zeros((s_q, h_q), DType::F32, q.device())?;
    let lse = Tensor::zeros((s_q, h_q), DType::F32, q.device())?;

    let (stream, device_id) = stream_and_device_id(q)?;
    let device = get_device_info(device_id)?;
    let q_stride = q.stride();
    let kv_stride = kv.stride();
    let indices_stride = indices.stride();
    let strides = SparsePrefillStrides {
        q_s_q: q_stride[0],
        q_h_q: q_stride[1],
        kv_s_kv: kv_stride[0],
        kv_h_kv: kv_stride[1],
        indices_s_q: indices_stride[0],
        indices_h_kv: indices_stride[1],
    };

    let params = SparsePrefillLaunchParams {
        dims,
        config,
        q: tensor_ptr_bf16(q, &stream, "q")?,
        kv: tensor_ptr_bf16(kv, &stream, "kv")?,
        indices: tensor_ptr_i32(indices, &stream, "indices")?,
        attn_sink: match attn_sink {
            Some(attn_sink) => tensor_ptr_f32(attn_sink, &stream, "attn_sink")?,
            None => std::ptr::null(),
        },
        topk_length: match topk_length {
            Some(topk_length) => tensor_ptr_i32(topk_length, &stream, "topk_length")?,
            None => std::ptr::null(),
        },
        strides,
        out: tensor_mut_ptr_bf16(&out, &stream, "out")?,
        max_logits: tensor_mut_ptr_f32(&max_logits, &stream, "max_logits")?,
        lse: tensor_mut_ptr_f32(&lse, &stream, "lse")?,
        num_sm: usize::try_from(device.num_sms)
            .map_err(|_| crate::Error::Tensor("num_sms overflow".to_string()))?,
        stream: stream.cu_stream() as *mut std::ffi::c_void,
    };

    unsafe { sparse_prefill_bf16(&params)? };

    Ok(SparsePrefillOutput {
        out,
        max_logits,
        lse,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use candle::{DType, Device, Tensor};

    use super::*;

    #[test]
    fn sparse_prefill_sm90_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q = Tensor::zeros((16, 64, 512), DType::BF16, &device)?;
        let kv = Tensor::zeros((128, 1, 512), DType::BF16, &device)?;
        let indices = Tensor::zeros((16, 1, 128), DType::I32, &device)?;
        let output = sparse_prefill(
            &q,
            &kv,
            &indices,
            None,
            None,
            SparsePrefillConfig {
                softmax_scale: 1.0,
                d_v: 512,
                pad_heads: false,
            },
        )?;

        assert_eq!(output.out.dims3()?, (16, 64, 512));
        assert_eq!(output.out.dtype(), DType::BF16);
        assert_eq!(output.max_logits.dims2()?, (16, 64));
        assert_eq!(output.lse.dims2()?, (16, 64));
        device.synchronize()?;

        Ok(())
    }
}

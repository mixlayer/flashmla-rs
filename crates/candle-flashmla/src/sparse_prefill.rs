//! Sparse prefill API re-exports for Candle integration.

pub use flashmla::{SparsePrefillConfig, SparsePrefillDims, SparsePrefillStrides};

use candle::{DType, Tensor};
use flashmla::{SparsePrefillLaunchParams, get_device_info, sparse_prefill_bf16};

use crate::{
    Result,
    error::invalid_arg,
    tensor::cuda::{
        ensure_dtype, ensure_last_dim_contiguous, ensure_rank, ensure_same_device,
        stream_and_device_id, tensor_ptr_bf16, tensor_ptr_f32, tensor_ptr_i32,
    },
};

/// Output tensors returned by sparse prefill.
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
    let (stream, device_id) = stream_and_device_id(q)?;
    let device = get_device_info(device_id)?;
    dims.validate_for_arch(device.arch)?;

    // Deliberately zero outputs for correctness diagnostics. FlashMLA is expected to overwrite
    // every element, so retained zeroes identify incomplete kernel coverage without exposing
    // allocator contents.
    let out = Tensor::zeros((s_q, h_q, config.d_v), DType::BF16, q.device())?;
    let max_logits = Tensor::zeros((s_q, h_q), DType::F32, q.device())?;
    let lse = Tensor::zeros((s_q, h_q), DType::F32, q.device())?;

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

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (kv_storage, kv_layout) = kv.storage_and_layout();
        let (indices_storage, indices_layout) = indices.storage_and_layout();
        let attn_sink_storage_and_layout = attn_sink.map(Tensor::storage_and_layout);
        let topk_length_storage_and_layout = topk_length.map(Tensor::storage_and_layout);
        let (out_storage, out_layout) = out.storage_and_layout();
        let (max_logits_storage, max_logits_layout) = max_logits.storage_and_layout();
        let (lse_storage, lse_layout) = lse.storage_and_layout();
        let q_ptr = tensor_ptr_bf16(&q_storage, q_layout.start_offset(), &stream, "q")?;
        let kv_ptr = tensor_ptr_bf16(&kv_storage, kv_layout.start_offset(), &stream, "kv")?;
        let indices_ptr = tensor_ptr_i32(
            &indices_storage,
            indices_layout.start_offset(),
            &stream,
            "indices",
        )?;
        let attn_sink_ptr = match &attn_sink_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_f32(
                storage,
                layout.start_offset(),
                &stream,
                "attn_sink",
            )?),
            None => None,
        };
        let topk_length_ptr = match &topk_length_storage_and_layout {
            Some((storage, layout)) => Some(tensor_ptr_i32(
                storage,
                layout.start_offset(),
                &stream,
                "topk_length",
            )?),
            None => None,
        };
        let out_ptr = tensor_ptr_bf16(&out_storage, out_layout.start_offset(), &stream, "out")?;
        let max_logits_ptr = tensor_ptr_f32(
            &max_logits_storage,
            max_logits_layout.start_offset(),
            &stream,
            "max_logits",
        )?;
        let lse_ptr = tensor_ptr_f32(&lse_storage, lse_layout.start_offset(), &stream, "lse")?;

        let params = SparsePrefillLaunchParams {
            dims,
            config,
            q: q_ptr.as_const_void(),
            kv: kv_ptr.as_const_void(),
            indices: indices_ptr.as_const_i32(),
            attn_sink: attn_sink_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_f32()),
            topk_length: topk_length_ptr
                .as_ref()
                .map_or(std::ptr::null(), |ptr| ptr.as_const_i32()),
            strides,
            out: out_ptr.as_mut_void(),
            max_logits: max_logits_ptr.as_mut_f32(),
            lse: lse_ptr.as_mut_f32(),
            num_sm: usize::try_from(device.num_sms)
                .map_err(|_| crate::Error::Tensor("num_sms overflow".to_string()))?,
            stream: stream.cu_stream() as *mut std::ffi::c_void,
        };

        unsafe { sparse_prefill_bf16(&params)? };
    }

    Ok(SparsePrefillOutput {
        out,
        max_logits,
        lse,
    })
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device, Tensor};
    use half::bf16;

    use super::*;

    #[test]
    #[ignore = "requires a visible SM90 CUDA GPU"]
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

    #[test]
    #[ignore = "requires a visible SM100 CUDA GPU"]
    fn sparse_prefill_sm100_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let info = get_device_info(0)?;
        assert_eq!(info.arch, flashmla::Arch::Sm100f);

        for (h_q, d_qk, topk) in [(64, 576, 64), (128, 512, 64), (128, 576, 128)] {
            let q = Tensor::from_vec(vec![bf16::ZERO; h_q * d_qk], (1, h_q, d_qk), &device)?;
            let kv = Tensor::from_vec(vec![bf16::ONE; topk * d_qk], (topk, 1, d_qk), &device)?;
            let indices = Tensor::from_vec(
                (0..i32::try_from(topk).unwrap()).collect::<Vec<_>>(),
                (1, 1, topk),
                &device,
            )?;
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
            device.synchronize()?;

            assert_eq!(output.out.dims3()?, (1, h_q, 512));
            let values = output.out.flatten_all()?.to_vec1::<bf16>()?;
            assert!(values.iter().all(|value| f32::from(*value).is_finite()));
            assert!(
                values
                    .iter()
                    .all(|value| (f32::from(*value) - 1.0).abs() <= 0.02)
            );
            let max_logits = output.max_logits.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                max_logits
                    .iter()
                    .all(|value| value.is_finite() && value.abs() <= 0.02)
            );
            let lse = output.lse.flatten_all()?.to_vec1::<f32>()?;
            let expected_lse = (topk as f32).ln();
            assert!(lse.iter().all(|value| value.is_finite()));
            assert!(lse.iter().all(|value| (value - expected_lse).abs() <= 0.02));
        }

        Ok(())
    }
}

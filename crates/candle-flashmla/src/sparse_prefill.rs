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
    dims.validate()?;

    // SAFETY: The FlashMLA prefill kernel fully overwrites `out`, `max_logits`, and `lse`
    // for every query/head element before these tensors are returned.
    let (out, max_logits, lse) = unsafe {
        (
            Tensor::empty((s_q, h_q, config.d_v), DType::BF16, q.device())?,
            Tensor::empty((s_q, h_q), DType::F32, q.device())?,
            Tensor::empty((s_q, h_q), DType::F32, q.device())?,
        )
    };

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

    use super::*;

    #[test]
    #[ignore = "requires a visible SM90 or SM100 CUDA GPU"]
    fn sparse_prefill_cuda_smoke() -> Result<()> {
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
    #[ignore = "requires a visible SM90 or SM100 CUDA GPU"]
    fn sparse_prefill_matches_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (s_q, s_kv, h_q, d_qk, topk) = (2, 160, 64, 512, 128);
        let q_values = (0..s_q * h_q * d_qk)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.01)
            .collect::<Vec<_>>();
        let kv_values = (0..s_kv * d_qk)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.0125)
            .collect::<Vec<_>>();
        let index_values = (0..s_q * topk)
            .map(|index| i32::try_from((index * 17 + 3) % s_kv).unwrap())
            .collect::<Vec<_>>();
        let topk_lengths = vec![97i32, 128];
        let sink_values = (0..h_q)
            .map(|head| (head as f32 - 31.5) * 0.005)
            .collect::<Vec<_>>();

        let q = Tensor::from_vec(q_values, (s_q, h_q, d_qk), &device)?.to_dtype(DType::BF16)?;
        let kv = Tensor::from_vec(kv_values, (s_kv, 1, d_qk), &device)?.to_dtype(DType::BF16)?;
        let indices = Tensor::from_vec(index_values.clone(), (s_q, 1, topk), &device)?;
        let topk_length = Tensor::from_vec(topk_lengths.clone(), s_q, &device)?;
        let attn_sink = Tensor::from_vec(sink_values.clone(), h_q, &device)?;
        let scale = (d_qk as f32).sqrt().recip();
        let output = sparse_prefill(
            &q,
            &kv,
            &indices,
            Some(&topk_length),
            Some(&attn_sink),
            SparsePrefillConfig {
                softmax_scale: scale,
                d_v: d_qk,
                pad_heads: false,
            },
        )?
        .out
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;

        let q = q.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let kv = kv.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        for query in 0..s_q {
            let valid_topk = usize::try_from(topk_lengths[query]).unwrap();
            for head in 0..h_q {
                let mut scores = Vec::with_capacity(valid_topk);
                for &index in &index_values[query * topk..query * topk + valid_topk] {
                    let index = usize::try_from(index).unwrap();
                    let dot = q[query][head]
                        .iter()
                        .zip(&kv[index][0])
                        .map(|(q, kv)| q * kv)
                        .sum::<f32>();
                    scores.push((index, dot * scale));
                }
                let max_score = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(sink_values[head], f32::max);
                let denominator = (sink_values[head] - max_score).exp()
                    + scores
                        .iter()
                        .map(|(_, score)| (*score - max_score).exp())
                        .sum::<f32>();
                for dim in 0..d_qk {
                    let expected = scores
                        .iter()
                        .map(|(index, score)| (*score - max_score).exp() * kv[*index][0][dim])
                        .sum::<f32>()
                        / denominator;
                    let error = (output[query][head][dim] - expected).abs();
                    assert!(
                        error <= 0.005,
                        "mismatch at query={query} head={head} dim={dim}: got {}, expected {expected}, error {error}",
                        output[query][head][dim]
                    );
                }
            }
        }

        Ok(())
    }
}

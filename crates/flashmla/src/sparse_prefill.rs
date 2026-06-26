use std::ffi::c_void;

use flashmla_sys::{
    cudaStream_t, flashmla_sparse_prefill_bf16 as sys_sparse_prefill_bf16,
    flashmla_sparse_prefill_params_t, flashmla_status_t,
};

use crate::{Error, Result};

/// Runtime options for sparse prefill.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SparsePrefillConfig {
    /// Softmax scale applied to QK logits.
    pub softmax_scale: f32,
    /// Value head dimension. FlashMLA sparse MLA currently requires `512`.
    pub d_v: usize,
    /// Whether higher-level integrations should pad query heads before launch.
    pub pad_heads: bool,
}

impl Default for SparsePrefillConfig {
    fn default() -> Self {
        Self {
            softmax_scale: 1.0,
            d_v: 512,
            pad_heads: true,
        }
    }
}

/// Shape parameters for sparse prefill.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparsePrefillDims {
    /// Query sequence length.
    pub s_q: usize,
    /// KV sequence length.
    pub s_kv: usize,
    /// Query head count after any required padding.
    pub h_q: usize,
    /// KV head count. Sparse MLA currently expects MQA-style `1`.
    pub h_kv: usize,
    /// Query/key head dimension. FlashMLA supports `512` and `576`.
    pub d_qk: usize,
    /// Value head dimension. FlashMLA sparse MLA currently requires `512`.
    pub d_v: usize,
    /// Number of sparse KV indices per query.
    pub topk: usize,
}

impl SparsePrefillDims {
    /// Validates architecture-independent sparse prefill shape constraints.
    pub fn validate(self) -> Result<()> {
        if self.s_q == 0 || self.s_kv == 0 || self.topk == 0 {
            return Err(Error::InvalidArgument(
                "s_q, s_kv, and topk must be non-zero".to_string(),
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
        Ok(())
    }

    fn validate_sm90(self) -> Result<()> {
        self.validate()?;
        if self.topk % 128 != 0 {
            return Err(Error::InvalidArgument(format!(
                "SM90 sparse prefill requires topk to be a multiple of 128, got {}",
                self.topk
            )));
        }
        Ok(())
    }
}

/// Element strides for raw sparse prefill tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparsePrefillStrides {
    /// Element stride between query sequence positions.
    pub q_s_q: usize,
    /// Element stride between query heads.
    pub q_h_q: usize,
    /// Element stride between KV sequence positions.
    pub kv_s_kv: usize,
    /// Element stride between KV heads.
    pub kv_h_kv: usize,
    /// Element stride between sparse-index query positions.
    pub indices_s_q: usize,
    /// Element stride between sparse-index KV heads.
    pub indices_h_kv: usize,
}

impl SparsePrefillStrides {
    /// Validates that all raw tensor strides are positive and fit in the C ABI.
    pub fn validate(self) -> Result<()> {
        checked_stride(self.q_s_q, "q_s_q")?;
        checked_stride(self.q_h_q, "q_h_q")?;
        checked_stride(self.kv_s_kv, "kv_s_kv")?;
        checked_stride(self.kv_h_kv, "kv_h_kv")?;
        checked_stride(self.indices_s_q, "indices_s_q")?;
        checked_stride(self.indices_h_kv, "indices_h_kv")?;
        Ok(())
    }
}

/// Raw pointer parameters for launching BF16 sparse prefill.
#[derive(Debug, Copy, Clone)]
pub struct SparsePrefillLaunchParams {
    /// Sparse prefill tensor dimensions.
    pub dims: SparsePrefillDims,
    /// Sparse prefill runtime options.
    pub config: SparsePrefillConfig,
    /// Raw BF16 query pointer with shape `[s_q, h_q, d_qk]`.
    pub q: *const c_void,
    /// Raw BF16 KV pointer with shape `[s_kv, h_kv, d_qk]`.
    pub kv: *const c_void,
    /// Raw I32 sparse index pointer with shape `[s_q, h_kv, topk]`.
    pub indices: *const i32,
    /// Optional raw F32 attention sink pointer with shape `[h_q]`.
    pub attn_sink: *const f32,
    /// Optional raw I32 top-k length pointer with shape `[s_q]`.
    pub topk_length: *const i32,
    /// Element strides for `q`, `kv`, and `indices`.
    pub strides: SparsePrefillStrides,
    /// Raw BF16 output pointer with shape `[s_q, h_q, d_v]`.
    pub out: *mut c_void,
    /// Raw F32 max-logits output pointer with shape `[s_q, h_q]`.
    pub max_logits: *mut f32,
    /// Raw F32 log-sum-exp output pointer with shape `[s_q, h_q]`.
    pub lse: *mut f32,
    /// Number of SMs to pass to the upstream kernel.
    pub num_sm: usize,
    /// CUDA stream used for the kernel launch.
    pub stream: cudaStream_t,
}

impl SparsePrefillLaunchParams {
    fn validate(self) -> Result<()> {
        self.dims.validate_sm90()?;
        self.strides.validate()?;
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
            || self.max_logits.is_null()
            || self.lse.is_null()
        {
            return Err(Error::InvalidArgument(
                "q, kv, indices, out, max_logits, and lse pointers must be non-null".to_string(),
            ));
        }
        if self.num_sm == 0 {
            return Err(Error::InvalidArgument(
                "num_sm must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    fn to_sys(self) -> Result<flashmla_sparse_prefill_params_t> {
        self.validate()?;
        Ok(flashmla_sparse_prefill_params_t {
            s_q: checked_i32(self.dims.s_q, "s_q")?,
            s_kv: checked_i32(self.dims.s_kv, "s_kv")?,
            h_q: checked_i32(self.dims.h_q, "h_q")?,
            h_kv: checked_i32(self.dims.h_kv, "h_kv")?,
            d_qk: checked_i32(self.dims.d_qk, "d_qk")?,
            d_v: checked_i32(self.dims.d_v, "d_v")?,
            topk: checked_i32(self.dims.topk, "topk")?,
            sm_scale: self.config.softmax_scale,
            q: self.q,
            kv: self.kv,
            indices: self.indices,
            attn_sink: self.attn_sink,
            topk_length: self.topk_length,
            stride_q_s_q: checked_stride(self.strides.q_s_q, "q_s_q")?,
            stride_q_h_q: checked_stride(self.strides.q_h_q, "q_h_q")?,
            stride_kv_s_kv: checked_stride(self.strides.kv_s_kv, "kv_s_kv")?,
            stride_kv_h_kv: checked_stride(self.strides.kv_h_kv, "kv_h_kv")?,
            stride_indices_s_q: checked_stride(self.strides.indices_s_q, "indices_s_q")?,
            stride_indices_h_kv: checked_stride(self.strides.indices_h_kv, "indices_h_kv")?,
            out: self.out,
            max_logits: self.max_logits,
            lse: self.lse,
            num_sm: checked_i32(self.num_sm, "num_sm")?,
            stream: self.stream,
        })
    }
}

/// Launches the SM90 BF16 sparse prefill kernel through `flashmla-sys`.
///
/// # Safety
///
/// All raw pointers in `params` must be valid CUDA device pointers for the documented shapes,
/// dtypes, and element strides. Output buffers must be writable and must not alias inputs in a way
/// that violates the upstream FlashMLA kernel requirements. `params.stream` must be a valid CUDA
/// stream for the current device, or null for the default stream.
pub unsafe fn sparse_prefill_bf16(params: &SparsePrefillLaunchParams) -> Result<()> {
    let sys_params = params.to_sys()?;
    let status = unsafe { sys_sparse_prefill_bf16(&sys_params) };
    if status == flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_status(status, "sparse prefill launch failed"))
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} does not fit in i32: {value}")))
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
        SparsePrefillDims {
            s_q: 16,
            s_kv: 128,
            h_q: 64,
            h_kv: 1,
            d_qk: 576,
            d_v: 512,
            topk: 32,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn sm90_validation_rejects_non_multiple_topk() {
        let dims = SparsePrefillDims {
            s_q: 16,
            s_kv: 128,
            h_q: 64,
            h_kv: 1,
            d_qk: 576,
            d_v: 512,
            topk: 32,
        };
        assert!(matches!(
            dims.validate_sm90(),
            Err(Error::InvalidArgument(_))
        ));
    }
}

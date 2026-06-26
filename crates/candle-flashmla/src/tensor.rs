use crate::{Error, Result};

/// Ensures a tensor-like shape has the expected rank.
pub fn ensure_rank(shape: &[usize], expected: usize) -> Result<()> {
    if shape.len() == expected {
        Ok(())
    } else {
        Err(Error::Tensor(format!(
            "expected rank {expected}, got rank {}",
            shape.len()
        )))
    }
}

/// Ensures two tensor-like shapes are identical.
pub fn ensure_same_shape(left: &[usize], right: &[usize]) -> Result<()> {
    if left == right {
        Ok(())
    } else {
        Err(Error::Tensor(format!(
            "shape mismatch: left={left:?}, right={right:?}"
        )))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use std::{ffi::c_void, sync::Arc};

    use candle::{
        DType, Storage, Tensor,
        cuda::{
            CudaDType,
            cudarc::driver::{CudaStream, DevicePtr, DeviceRepr},
        },
    };

    use crate::{
        Result,
        error::{Error, invalid_arg},
    };

    pub(crate) fn ensure_rank(t: &Tensor, rank: usize, name: &str) -> Result<()> {
        if t.rank() == rank {
            Ok(())
        } else {
            invalid_arg(format!("{name} must have rank {rank}, got {}", t.rank()))
        }
    }

    pub(crate) fn ensure_last_dim_contiguous(t: &Tensor, name: &str) -> Result<()> {
        match t.stride().last() {
            Some(1) => Ok(()),
            Some(stride) => invalid_arg(format!(
                "{name} must have a contiguous last dimension, got stride {stride}"
            )),
            None => invalid_arg(format!("{name} must have rank > 0")),
        }
    }

    pub(crate) fn ensure_same_device(reference: &Tensor, t: &Tensor, name: &str) -> Result<()> {
        if reference.device().same_device(t.device()) {
            Ok(())
        } else {
            invalid_arg(format!("{name} must be on the same device as q"))
        }
    }

    pub(crate) fn ensure_dtype(t: &Tensor, dtype: DType, name: &str) -> Result<()> {
        if t.dtype() == dtype {
            Ok(())
        } else {
            invalid_arg(format!(
                "{name} must have dtype {dtype:?}, got {:?}",
                t.dtype()
            ))
        }
    }

    pub(crate) fn stream_and_device_id(t: &Tensor) -> Result<(Arc<CudaStream>, i32)> {
        let cuda_dev = t.device().as_cuda_device()?;
        let stream = cuda_dev.cuda_stream();
        let device_id = i32::try_from(stream.context().ordinal())
            .map_err(|_| Error::Tensor("device id overflow".to_string()))?;
        Ok((stream, device_id))
    }

    pub(crate) fn tensor_ptr_bf16(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*const c_void> {
        tensor_ptr_typed::<half::bf16>(t, stream, name)
    }

    pub(crate) fn tensor_mut_ptr_bf16(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*mut c_void> {
        Ok(tensor_ptr_typed::<half::bf16>(t, stream, name)? as *mut c_void)
    }

    pub(crate) fn tensor_ptr_i32(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*const i32> {
        Ok(tensor_ptr_typed::<i32>(t, stream, name)? as *const i32)
    }

    pub(crate) fn tensor_ptr_f32(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*const f32> {
        Ok(tensor_ptr_typed::<f32>(t, stream, name)? as *const f32)
    }

    pub(crate) fn tensor_mut_ptr_i32(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*mut i32> {
        Ok(tensor_ptr_typed::<i32>(t, stream, name)? as *mut i32)
    }

    pub(crate) fn tensor_mut_ptr_f32(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*mut f32> {
        Ok(tensor_ptr_typed::<f32>(t, stream, name)? as *mut f32)
    }

    pub(crate) fn tensor_ptr_u8_or_f8(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*const c_void> {
        match t.dtype() {
            DType::U8 => tensor_ptr_typed::<u8>(t, stream, name),
            DType::F8E4M3 => tensor_ptr_typed::<float8::F8E4M3>(t, stream, name),
            dtype => invalid_arg(format!(
                "{name} must have dtype U8 or F8E4M3, got {dtype:?}"
            )),
        }
    }

    fn tensor_ptr_typed<T: CudaDType + DeviceRepr>(
        t: &Tensor,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<*const c_void> {
        let (storage, layout) = t.storage_and_layout();
        let ptr = match &*storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?.device_ptr(stream).0,
            _ => return invalid_arg(format!("{name} must be a CUDA tensor")),
        };
        let offset_bytes = layout
            .start_offset()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::Tensor(format!("{name} start_offset overflow")))?;
        let ptr = ptr
            .checked_add(offset_bytes as u64)
            .ok_or_else(|| Error::Tensor(format!("{name} pointer overflow")))?;
        Ok(ptr as usize as *const c_void)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_validation_reports_mismatch() {
        assert!(ensure_rank(&[1, 2, 3], 4).is_err());
    }
}

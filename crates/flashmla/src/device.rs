use flashmla_sys::{flashmla_get_device_info, flashmla_status_t};

use crate::{Arch, Error, Result};

/// CUDA device metadata needed for FlashMLA dispatch and workspace sizing.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// CUDA device ordinal.
    pub device_id: i32,
    /// CUDA compute capability major version.
    pub major: u32,
    /// CUDA compute capability minor version.
    pub minor: u32,
    /// Number of streaming multiprocessors reported by CUDA.
    pub num_sms: u32,
    /// FlashMLA-supported architecture for this device.
    pub arch: Arch,
}

/// Queries CUDA device properties and maps the compute capability to a FlashMLA architecture.
pub fn get_device_info(device_id: i32) -> Result<DeviceInfo> {
    if device_id < 0 {
        return Err(Error::InvalidArgument(
            "device_id must be non-negative".to_string(),
        ));
    }

    let mut major = 0;
    let mut minor = 0;
    let mut num_sms = 0;
    let status =
        unsafe { flashmla_get_device_info(device_id, &mut major, &mut minor, &mut num_sms) };
    if status != flashmla_status_t::FLASHMLA_STATUS_SUCCESS {
        return Err(Error::from_status(
            status,
            format!("failed to query CUDA device {device_id}"),
        ));
    }

    let major = u32::try_from(major)
        .map_err(|_| Error::Internal(format!("CUDA returned a negative major version: {major}")))?;
    let minor = u32::try_from(minor)
        .map_err(|_| Error::Internal(format!("CUDA returned a negative minor version: {minor}")))?;
    let num_sms = u32::try_from(num_sms)
        .map_err(|_| Error::Internal(format!("CUDA returned a negative SM count: {num_sms}")))?;
    let arch = Arch::from_compute_capability(major, minor)?;

    Ok(DeviceInfo {
        device_id,
        major,
        minor,
        num_sms,
        arch,
    })
}

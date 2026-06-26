use std::fmt;

use flashmla_sys::{flashmla_last_error, flashmla_status_t};

/// Result type used by the FlashMLA safe wrapper crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by FlashMLA safe wrapper operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A caller-provided parameter was invalid.
    InvalidArgument(String),
    /// The current GPU architecture is not supported by the requested FlashMLA path.
    UnsupportedArch(String),
    /// CUDA returned an error.
    Cuda(String),
    /// FlashMLA or the wrapper hit an unexpected internal error.
    Internal(String),
    /// A raw FFI status value could not be mapped.
    UnknownStatus(i32),
}

impl Error {
    /// Converts a raw FlashMLA status into a safe wrapper error.
    pub fn from_status(status: flashmla_status_t, context: impl Into<String>) -> Self {
        let context = append_last_error(context.into());
        match status {
            flashmla_status_t::FLASHMLA_STATUS_SUCCESS => {
                Error::Internal(format!("{context}: unexpected success status"))
            }
            flashmla_status_t::FLASHMLA_STATUS_INVALID_ARGUMENT => Error::InvalidArgument(context),
            flashmla_status_t::FLASHMLA_STATUS_UNSUPPORTED_ARCH => Error::UnsupportedArch(context),
            flashmla_status_t::FLASHMLA_STATUS_CUDA_ERROR => Error::Cuda(context),
            flashmla_status_t::FLASHMLA_STATUS_INTERNAL_ERROR => Error::Internal(context),
        }
    }
}

fn append_last_error(context: String) -> String {
    let message = unsafe {
        let ptr = flashmla_last_error();
        if ptr.is_null() {
            None
        } else {
            std::ffi::CStr::from_ptr(ptr).to_str().ok()
        }
    };

    match message {
        Some(message) if !message.is_empty() => format!("{context}: {message}"),
        _ => context,
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Error::UnsupportedArch(message) => write!(f, "unsupported architecture: {message}"),
            Error::Cuda(message) => write!(f, "cuda error: {message}"),
            Error::Internal(message) => write!(f, "internal error: {message}"),
            Error::UnknownStatus(status) => write!(f, "unknown flashmla status: {status}"),
        }
    }
}

impl std::error::Error for Error {}

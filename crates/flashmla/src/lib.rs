//! Candle-independent safe wrappers for FlashMLA raw FFI.

/// Architecture detection and target metadata.
pub mod arch;
/// CUDA device queries.
pub mod device;
/// Error types returned by the safe wrapper layer.
pub mod error;
/// Sparse decode parameter validation.
pub mod sparse_decode;
/// Sparse prefill parameter validation and raw launch wrappers.
pub mod sparse_prefill;
/// Workspace sizing metadata shared by wrapper APIs.
pub mod workspace;

pub use arch::Arch;
pub use device::{DeviceInfo, get_device_info};
pub use error::{Error, Result};
pub use sparse_decode::{
    SparseDecodeConfig, SparseDecodeDims, SparseDecodeLaunchParams, SparseDecodePlanMeta,
    SparseDecodePlanParams, SparseDecodeStrides, sparse_decode_bf16_fp8, sparse_decode_plan,
};
pub use sparse_prefill::{
    SparsePrefillConfig, SparsePrefillDims, SparsePrefillLaunchParams, SparsePrefillStrides,
    sparse_prefill_bf16,
};
pub use workspace::WorkspaceLayout;

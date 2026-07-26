//! Candle integration layer for FlashMLA.

/// Candle dense decode integration.
pub mod dense_decode;
/// Error types returned by Candle integration APIs.
pub mod error;
/// Candle sparse decode integration.
pub mod sparse_decode;
/// Candle sparse prefill integration.
pub mod sparse_prefill;
/// Candle tensor validation and pointer extraction helpers.
pub mod tensor;
/// Candle workspace allocation helpers.
pub mod workspace;

/// Candle integration error and result types.
pub use error::{Error, Result};

pub use dense_decode::{DenseDecodeOutput, DenseDecodePlan, dense_decode, dense_decode_plan};
pub use sparse_decode::{SparseDecodeOutput, SparseDecodePlan, sparse_decode, sparse_decode_plan};
pub use sparse_prefill::{SparsePrefillOutput, sparse_prefill};

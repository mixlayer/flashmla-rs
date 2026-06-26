use std::fmt;

/// Result type used by the Candle FlashMLA integration crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by Candle FlashMLA integration APIs.
#[derive(Debug)]
pub enum Error {
    /// Error returned by the Candle-independent FlashMLA wrapper.
    FlashMla(flashmla::Error),
    /// Error returned by Candle.
    Candle(candle::Error),
    /// Tensor validation or pointer extraction failed.
    Tensor(String),
}

impl From<flashmla::Error> for Error {
    fn from(error: flashmla::Error) -> Self {
        Self::FlashMla(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FlashMla(error) => write!(f, "{error}"),
            Error::Candle(error) => write!(f, "{error}"),
            Error::Tensor(message) => write!(f, "tensor error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle::Error> for Error {
    fn from(error: candle::Error) -> Self {
        Self::Candle(error)
    }
}

pub(crate) fn invalid_arg<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Tensor(message.into()))
}

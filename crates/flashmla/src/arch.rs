use crate::{Error, Result};

/// FlashMLA-supported NVIDIA GPU architecture.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Arch {
    /// Hopper datacenter GPUs compiled with `sm_90a`.
    Sm90a,
    /// Blackwell datacenter GPUs compiled with `sm_100f`.
    Sm100f,
}

impl Arch {
    /// Maps CUDA compute capability to a FlashMLA-supported architecture.
    pub fn from_compute_capability(major: u32, minor: u32) -> Result<Self> {
        match (major, minor) {
            (9, 0) => Ok(Self::Sm90a),
            (10, 0) => Ok(Self::Sm100f),
            (12, 0 | 1) => Err(Error::UnsupportedArch(
                "SM120/SM121 are intentionally unsupported by upstream FlashMLA".to_string(),
            )),
            _ => Err(Error::UnsupportedArch(format!(
                "expected SM90 or SM100, got compute capability {major}.{minor}"
            ))),
        }
    }

    /// Returns the NVCC architecture string used by FlashMLA for this architecture.
    pub fn nvcc_arch(self) -> &'static str {
        match self {
            Self::Sm90a => "sm_90a",
            Self::Sm100f => "sm_100f",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_supported_architectures() {
        assert_eq!(Arch::from_compute_capability(9, 0).unwrap(), Arch::Sm90a);
        assert_eq!(Arch::from_compute_capability(10, 0).unwrap(), Arch::Sm100f);
    }

    #[test]
    fn rejects_sm120_series() {
        assert!(matches!(
            Arch::from_compute_capability(12, 1),
            Err(Error::UnsupportedArch(_))
        ));
    }
}

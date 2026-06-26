/// Byte-size description for caller-owned FlashMLA workspace memory.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct WorkspaceLayout {
    /// Required workspace bytes.
    pub bytes: usize,
}

impl WorkspaceLayout {
    /// Creates a workspace layout with the specified byte size.
    pub fn new(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Returns true when no workspace allocation is required.
    pub fn is_empty(self) -> bool {
        self.bytes == 0
    }
}

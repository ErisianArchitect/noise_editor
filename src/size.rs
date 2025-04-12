#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }

    #[inline]
    pub const fn index(self, x: u32, y: u32) -> u32 {
        (y * self.width) + x
    }

    /// Gets the position based on the index.
    #[inline]
    pub const fn inv_index(self, index: u32) -> (u32, u32) {
        (index % self.width, index / self.width)
    }

    #[inline]
    pub const fn iter_width(self) -> std::ops::Range<u32> {
        0..self.width
    }

    #[inline]
    pub const fn iter_height(self) -> std::ops::Range<u32> {
        0..self.height
    }
}
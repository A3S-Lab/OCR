use super::wired::PixelRect;

/// Deterministic crop orientation selected from source-backed wire geometry.
///
/// SLANet-Plus expects table text in reading orientation. A scanned page may
/// contain a quarter-turned table even when its container reports no rotation,
/// so inference uses an oriented crop and maps every decoded point back to the
/// immutable source canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TableCropOrientation {
    Upright,
    Rotate90,
}

impl TableCropOrientation {
    pub(super) fn from_grid(
        region: PixelRect,
        horizontal_line_count: usize,
        vertical_line_count: usize,
    ) -> Self {
        let not_landscape = region.height >= region.width;
        let transposed_grid = vertical_line_count >= horizontal_line_count.saturating_mul(2)
            && horizontal_line_count >= 2;
        if not_landscape && transposed_grid {
            Self::Rotate90
        } else {
            Self::Upright
        }
    }

    pub(super) fn oriented_dimensions(self, region: PixelRect) -> (u32, u32) {
        match self {
            Self::Upright => (region.width, region.height),
            Self::Rotate90 => (region.height, region.width),
        }
    }

    pub(super) fn source_pixel(
        self,
        region: PixelRect,
        oriented_x: u32,
        oriented_y: u32,
    ) -> (u32, u32) {
        match self {
            Self::Upright => (region.x + oriented_x, region.y + oriented_y),
            Self::Rotate90 => (
                region.x + oriented_y,
                region.y + region.height.saturating_sub(1).saturating_sub(oriented_x),
            ),
        }
    }

    pub(super) fn source_boundary_point(
        self,
        region: PixelRect,
        oriented_x: u32,
        oriented_y: u32,
    ) -> (u32, u32) {
        match self {
            Self::Upright => (region.x + oriented_x, region.y + oriented_y),
            Self::Rotate90 => (
                region.x + oriented_y,
                region.y + region.height.saturating_sub(oriented_x),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGION: PixelRect = PixelRect {
        x: 10,
        y: 20,
        width: 200,
        height: 400,
    };

    #[test]
    fn only_a_non_landscape_transposed_grid_is_rotated() {
        assert_eq!(
            TableCropOrientation::from_grid(REGION, 5, 14),
            TableCropOrientation::Rotate90
        );
        assert_eq!(
            TableCropOrientation::from_grid(REGION, 14, 5),
            TableCropOrientation::Upright
        );
        assert_eq!(
            TableCropOrientation::from_grid(
                PixelRect {
                    width: 401,
                    height: 400,
                    ..REGION
                },
                5,
                14,
            ),
            TableCropOrientation::Upright
        );
    }

    #[test]
    fn rotate_90_maps_oriented_bounds_back_to_the_source_crop() {
        let orientation = TableCropOrientation::Rotate90;
        assert_eq!(orientation.oriented_dimensions(REGION), (400, 200));
        assert_eq!(orientation.source_pixel(REGION, 0, 0), (10, 419));
        assert_eq!(orientation.source_pixel(REGION, 399, 199), (209, 20));
        assert_eq!(orientation.source_boundary_point(REGION, 0, 0), (10, 420));
        assert_eq!(
            orientation.source_boundary_point(REGION, 400, 200),
            (210, 20)
        );
    }
}

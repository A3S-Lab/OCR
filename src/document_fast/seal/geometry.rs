use super::super::wired::PixelRect;

pub(super) fn intersection_over_union(left: PixelRect, right: PixelRect) -> f32 {
    let (intersection, left_area, right_area) = intersection_and_areas(left, right);
    if intersection == 0 {
        return 0.0;
    }
    intersection as f32 / (left_area + right_area - intersection) as f32
}

pub(super) fn intersection_over_smaller(left: PixelRect, right: PixelRect) -> f32 {
    let (intersection, left_area, right_area) = intersection_and_areas(left, right);
    let smaller = left_area.min(right_area);
    if smaller == 0 {
        0.0
    } else {
        intersection as f32 / smaller as f32
    }
}

pub(super) fn center_is_inside(inner: PixelRect, outer: PixelRect) -> bool {
    let center_x_twice = u64::from(inner.x)
        .saturating_mul(2)
        .saturating_add(u64::from(inner.width));
    let center_y_twice = u64::from(inner.y)
        .saturating_mul(2)
        .saturating_add(u64::from(inner.height));
    let left_twice = u64::from(outer.x).saturating_mul(2);
    let top_twice = u64::from(outer.y).saturating_mul(2);
    let right_twice = u64::from(outer.x.saturating_add(outer.width)).saturating_mul(2);
    let bottom_twice = u64::from(outer.y.saturating_add(outer.height)).saturating_mul(2);
    center_x_twice >= left_twice
        && center_x_twice <= right_twice
        && center_y_twice >= top_twice
        && center_y_twice <= bottom_twice
}

pub(super) fn area(region: PixelRect) -> u64 {
    u64::from(region.width).saturating_mul(u64::from(region.height))
}

pub(super) fn intersection_and_areas(left: PixelRect, right: PixelRect) -> (u64, u64, u64) {
    let left_x2 = left.x.saturating_add(left.width);
    let left_y2 = left.y.saturating_add(left.height);
    let right_x2 = right.x.saturating_add(right.width);
    let right_y2 = right.y.saturating_add(right.height);
    let intersection_width = left_x2.min(right_x2).saturating_sub(left.x.max(right.x));
    let intersection_height = left_y2.min(right_y2).saturating_sub(left.y.max(right.y));
    let intersection = u64::from(intersection_width).saturating_mul(u64::from(intersection_height));
    (intersection, area(left), area(right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_metrics_preserve_instance_direction() {
        let inner = PixelRect {
            x: 20,
            y: 20,
            width: 20,
            height: 20,
        };
        let outer = PixelRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        assert_eq!(intersection_over_smaller(inner, outer), 1.0);
        assert_eq!(intersection_over_union(inner, outer), 0.04);
        assert!(center_is_inside(inner, outer));
        assert!(!center_is_inside(outer, inner));
    }
}

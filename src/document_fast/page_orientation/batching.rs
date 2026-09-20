use a3s_power::inference::InferenceLimits;
use a3s_use_core::UseResult;

use super::{orientation_error, preprocess};

pub(super) fn maximum_batch_size(
    limits: &InferenceLimits,
    requested_cap: usize,
) -> UseResult<usize> {
    let input_bytes_per_image = preprocess::INPUT_ELEMENTS_PER_IMAGE
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| orientation_error("Page-orientation input bytes overflowed."))?;
    let input_limit = limits.max_input_bytes / input_bytes_per_image;
    let tensor_limit = limits.max_tensor_elements / preprocess::INPUT_ELEMENTS_PER_IMAGE;
    let maximum = requested_cap
        .min(crate::batch::MAX_BATCH_SLOTS)
        .min(input_limit)
        .min(tensor_limit);
    if maximum == 0 {
        return Err(orientation_error(
            "Power limits cannot admit one page-orientation tensor.",
        ));
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_is_derived_from_power_and_protocol_limits() {
        let limits = InferenceLimits::default();
        let expected = limits
            .max_input_bytes
            .checked_div(preprocess::INPUT_ELEMENTS_PER_IMAGE * std::mem::size_of::<f32>())
            .unwrap()
            .min(crate::batch::MAX_BATCH_SLOTS);

        assert_eq!(
            maximum_batch_size(&limits, crate::batch::MAX_BATCH_SLOTS).unwrap(),
            expected
        );
        assert_eq!(maximum_batch_size(&limits, 8).unwrap(), 8);
    }

    #[test]
    fn limits_must_admit_at_least_one_orientation_tensor() {
        let limits = InferenceLimits {
            max_input_bytes: 1,
            ..InferenceLimits::default()
        };

        assert_eq!(
            maximum_batch_size(&limits, crate::batch::MAX_BATCH_SLOTS)
                .unwrap_err()
                .code,
            "use.ocr.orientation_failed"
        );
    }
}

use crate::config::ModelProfile;

const PAGE_ORIENTATION_MODEL: &str = "pp-lcnet-x1-doc-orientation";
const DOCUMENT_STRUCTURE_MODEL: &str = "source-wired-grid+slanet-plus-fallback";
const DOCUMENT_LAYOUT_MODEL: &str = "pp-doclayout-s";
const SEAL_TEXT_MODEL: &str = "pp-ocr-v4-mobile-seal-det";

pub(super) fn document_fast_model_name(
    text_profile: ModelProfile,
    orientation: bool,
    document_layout: bool,
    seal: Option<(&str, bool)>,
) -> String {
    let mut name = String::new();
    if orientation {
        name.push_str(PAGE_ORIENTATION_MODEL);
        name.push('+');
    }
    name.push_str(text_profile.execution_family());
    name.push('+');
    name.push_str(DOCUMENT_STRUCTURE_MODEL);
    if document_layout {
        name.push('+');
        name.push_str(DOCUMENT_LAYOUT_MODEL);
    }
    if let Some((layout_family, seal_text)) = seal {
        name.push('+');
        name.push_str(layout_family);
        if seal_text {
            name.push('+');
            name.push_str(SEAL_TEXT_MODEL);
        }
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelProfile, ModelVariant};

    const SMALL: ModelProfile = ModelProfile::new(ModelVariant::Small, ModelVariant::Small);
    const HYBRID: ModelProfile = ModelProfile::new(ModelVariant::Small, ModelVariant::Tiny);

    #[test]
    fn model_identity_names_only_the_exact_composed_stages() {
        assert_eq!(
            document_fast_model_name(SMALL, false, false, None),
            "pp-ocr-v6-small+source-wired-grid+slanet-plus-fallback"
        );
        assert_eq!(
            document_fast_model_name(SMALL, true, false, None),
            "pp-lcnet-x1-doc-orientation+pp-ocr-v6-small+source-wired-grid+slanet-plus-fallback"
        );
        assert_eq!(
            document_fast_model_name(SMALL, false, false, Some(("picodet-s-layout-3cls", false)),),
            "pp-ocr-v6-small+source-wired-grid+slanet-plus-fallback+picodet-s-layout-3cls"
        );
        assert_eq!(
            document_fast_model_name(
                SMALL,
                true,
                true,
                Some(("picodet-l-layout-3cls", true)),
            ),
            "pp-lcnet-x1-doc-orientation+pp-ocr-v6-small+source-wired-grid+slanet-plus-fallback+pp-doclayout-s+picodet-l-layout-3cls+pp-ocr-v4-mobile-seal-det"
        );
    }

    #[test]
    fn model_identity_distinguishes_a_mixed_text_profile() {
        assert_eq!(
            document_fast_model_name(HYBRID, false, false, None),
            "pp-ocr-v6-small-detection-tiny-recognition+source-wired-grid+slanet-plus-fallback"
        );
        assert_ne!(
            document_fast_model_name(HYBRID, false, false, None),
            document_fast_model_name(SMALL, false, false, None)
        );
    }
}

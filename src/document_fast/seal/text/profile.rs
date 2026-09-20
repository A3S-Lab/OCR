use crate::config::{DetectionConfig, ModelVariant};

pub(super) const FAMILY: &str = "pp-ocr-v4-mobile-seal-det";
pub(super) const ROLE: &str = "seal-text-detection";
pub(super) const REVISION: &str = "paddlex-paddle3.0.0";
pub(super) const SOURCE_GRAPH_SHA256: &str =
    "39854c9489b4cb0c5f47ff361b31e966d56b791b92a7ac99686f8a1a2fd8e5c1";
pub(super) const GRAPH_SHA256: &str =
    "2e9d197579095f5816dbf8c2dd69514e3676b95bf87eaa103de0c04f4d31fbae";
pub(super) const WEIGHTS_FILE_SHA256: &str =
    "22d45b9b894c9cc5eefcd52dc874448004f307297f727adad80d110e1a457e23";
pub(super) const WEIGHTS_COLLECTION_SHA256: &str =
    "87a2ca81a27051c17ca5aad60f05cbc161fa0def010c983d435028a1916259db";
pub(super) const WEIGHTS_BYTES: u64 = 4_709_036;
pub(super) const RESIZE_LONG: u32 = 736;
pub(super) const RESIZE_STRIDE: u32 = 128;

pub(super) fn detection_config() -> DetectionConfig {
    DetectionConfig {
        model_variant: ModelVariant::Small,
        scale: 1.0 / 255.0,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        threshold: 0.2,
        box_threshold: 0.6,
        max_candidates: 1_000,
        unclip_ratio: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_preprocess_and_db_contract_is_explicit() {
        let config = detection_config();
        assert_eq!(RESIZE_LONG, 736);
        assert_eq!(RESIZE_STRIDE, 128);
        assert_eq!(config.threshold, 0.2);
        assert_eq!(config.box_threshold, 0.6);
        assert_eq!(config.unclip_ratio, 0.5);
        assert_eq!(config.max_candidates, 1_000);
    }
}

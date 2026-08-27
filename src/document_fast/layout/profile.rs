use crate::OcrLayoutRole;

pub(super) const FAMILY: &str = "pp-doclayout-s";
pub(super) const REVISION: &str = "paddleocr-paddle3-reviewed-v1";
pub(super) const GRAPH_ROLE: &str = "layout-raw-head";
pub(super) const SOURCE_GRAPH_SHA256: &str =
    "ac09e931895d4c442e5379ab3b7e9b583baf288e816ab7675ca519da9eb2a9d7";
pub(super) const GRAPH_SHA256: &str =
    "557be4760636da59c32c9baf9cf73496390e61c365c6e54f1d861066afb00ab7";
pub(super) const WEIGHTS_FILE_SHA256: &str =
    "f17e00dd97decb9b1446a4273b85913fb230bd0c2fd54ae5696009db6abfa62c";
pub(super) const WEIGHTS_COLLECTION_SHA256: &str =
    "3ef78a4d2f302f1329072f87535888099ce890091e005e968cbdc5b8d226fff4";
pub(super) const WEIGHTS_BYTES: u64 = 4_841_248;
pub(super) const GRAPH_OPSET: u32 = 3;

pub(super) const INPUT_SIDE: usize = 480;
pub(super) const INPUT_ELEMENTS_PER_IMAGE: usize = 3 * INPUT_SIDE * INPUT_SIDE;
pub(super) const LOCATION_COUNT: usize = 4_789;
pub(super) const CLASS_COUNT: usize = 23;
pub(super) const OUTPUT_WIDTH: usize = 4 + CLASS_COUNT;
pub(super) const SCORE_THRESHOLD: f32 = 0.3;
pub(super) const NMS_IOU_THRESHOLD: f32 = 0.5;
pub(super) const NMS_TOP_K: usize = 1_000;
pub(super) const KEEP_TOP_K: usize = 100;
pub(super) const MAX_BATCH_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LayoutClass {
    pub(super) raw_label: &'static str,
    pub(super) role: OcrLayoutRole,
}

pub(super) const CLASSES: [LayoutClass; CLASS_COUNT] = [
    LayoutClass {
        raw_label: "paragraph_title",
        role: OcrLayoutRole::Heading,
    },
    LayoutClass {
        raw_label: "image",
        role: OcrLayoutRole::Figure,
    },
    LayoutClass {
        raw_label: "text",
        role: OcrLayoutRole::Text,
    },
    LayoutClass {
        raw_label: "number",
        role: OcrLayoutRole::Text,
    },
    LayoutClass {
        raw_label: "abstract",
        role: OcrLayoutRole::Paragraph,
    },
    LayoutClass {
        raw_label: "content",
        role: OcrLayoutRole::Paragraph,
    },
    LayoutClass {
        raw_label: "figure_title",
        role: OcrLayoutRole::Caption,
    },
    LayoutClass {
        raw_label: "formula",
        role: OcrLayoutRole::EquationBlock,
    },
    LayoutClass {
        raw_label: "table",
        role: OcrLayoutRole::Table,
    },
    LayoutClass {
        raw_label: "table_title",
        role: OcrLayoutRole::Caption,
    },
    LayoutClass {
        raw_label: "reference",
        role: OcrLayoutRole::Paragraph,
    },
    LayoutClass {
        raw_label: "doc_title",
        role: OcrLayoutRole::Title,
    },
    LayoutClass {
        raw_label: "footnote",
        role: OcrLayoutRole::Footnote,
    },
    LayoutClass {
        raw_label: "header",
        role: OcrLayoutRole::Header,
    },
    LayoutClass {
        raw_label: "algorithm",
        role: OcrLayoutRole::CodeBlock,
    },
    LayoutClass {
        raw_label: "footer",
        role: OcrLayoutRole::Footer,
    },
    LayoutClass {
        raw_label: "seal",
        role: OcrLayoutRole::Seal,
    },
    LayoutClass {
        raw_label: "chart_title",
        role: OcrLayoutRole::Caption,
    },
    LayoutClass {
        raw_label: "chart",
        role: OcrLayoutRole::Chart,
    },
    LayoutClass {
        raw_label: "formula_number",
        role: OcrLayoutRole::Text,
    },
    LayoutClass {
        raw_label: "header_image",
        role: OcrLayoutRole::HeaderFigure,
    },
    LayoutClass {
        raw_label: "footer_image",
        role: OcrLayoutRole::FooterFigure,
    },
    LayoutClass {
        raw_label: "aside_text",
        role: OcrLayoutRole::Note,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_taxonomy_is_index_bound_and_complete() {
        assert_eq!(CLASSES.len(), CLASS_COUNT);
        assert_eq!(CLASSES[0].raw_label, "paragraph_title");
        assert_eq!(CLASSES[11].role, OcrLayoutRole::Title);
        assert_eq!(CLASSES[16].role, OcrLayoutRole::Seal);
        assert_eq!(CLASSES[22].raw_label, "aside_text");
        assert_eq!(OUTPUT_WIDTH, 27);
    }
}

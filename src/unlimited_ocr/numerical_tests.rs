use std::time::Instant;

use image::ImageEncoder as _;
use sha2::{Digest, Sha256};

use super::*;

const NUMERICAL_PARITY_SOURCE_SHA256: &str =
    "4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4";
const NUMERICAL_PARITY_CROP_RGB_SHA256: &str =
    "9221e07746854f2fb12ded3a760bf692314612709eff208731cf85d771fb208d";
const NUMERICAL_PARITY_EXACT_PREFIX: usize = 15;
const NUMERICAL_PARITY_MAX_UNSTABLE_STEPS: usize = 2;
const NUMERICAL_PARITY_MAX_EXPECTED_RANK: usize = 2;
const NUMERICAL_PARITY_MAX_LOGIT_DELTA: f32 = 0.25;
const NUMERICAL_PARITY_TOKENS: &[u32] = &[
    128_818, 16_771, 764, 2_116, 14, 223, 18, 14, 223, 17_804, 14, 223, 3_186, 63, 128_819, 2_193,
    223, 2_069, 20_379, 49_447, 201, 128_818, 10_212, 764, 2_116, 14, 223, 2_170, 14, 223, 8_834,
    14, 223, 9_775, 63, 128_819, 5_584, 1_147, 4_677, 126_316, 4_951, 17_169, 201, 128_818, 2_067,
    764, 18, 14, 223, 2_722, 14, 223, 8_834, 14, 223, 14_972, 63, 128_819, 12_973, 23_719, 223,
    14_590, 70_380, 223,
];
const NUMERICAL_PARITY_TEXT: &str = concat!(
    "<|det|>header [39, 0, 307, 67]<|/det|>",
    ".com 中国收藏热线\n",
    "<|det|>title [39, 48, 999, 144]<|/det|>",
    "登机牌 BOARDING PA\n",
    "<|det|>text [0, 196, 999, 245]<|/det|>",
    "FLIGHT 日期 DATE",
);

#[test]
#[ignore = "requires the pinned 6.7 GiB checkpoint and numerical parity source image"]
fn official_numerical_stability_and_grounding_match_upstream() {
    let root = std::env::var_os("A3S_UNLIMITED_OCR_MODEL_DIR")
        .expect("A3S_UNLIMITED_OCR_MODEL_DIR must name the pinned official checkpoint");
    let image_path = std::env::var_os("A3S_UNLIMITED_OCR_PARITY_IMAGE")
        .expect("A3S_UNLIMITED_OCR_PARITY_IMAGE must name the pinned parity source image");
    let source =
        std::fs::read(&image_path).expect("the pinned parity source image must be readable");
    assert_eq!(
        format!("{:x}", Sha256::digest(&source)),
        NUMERICAL_PARITY_SOURCE_SHA256
    );
    let bytes = numerical_parity_crop(&source);

    let started = Instant::now();
    let config = UnlimitedOcrConfig::new(root)
        .unwrap()
        .with_max_generated_tokens(NUMERICAL_PARITY_TOKENS.len())
        .unwrap();
    let session = UnlimitedOcrSession::load(&config).unwrap();
    let loaded = started.elapsed();
    let cancellation = CancellationToken::new();
    let permit = session.runtime.begin(&cancellation).unwrap();
    let image = preprocess(&bytes, &session.limits, &cancellation).unwrap();
    let prompt = session.tokenizer.encode_prompt(&image).unwrap();
    assert_eq!(prompt.token_ids.len(), 277);
    let preprocessed = started.elapsed();
    let vision = session
        .vision
        .encode(&image, &permit, &cancellation)
        .unwrap();
    let encoded = started.elapsed();
    let scores = session
        .decoder
        .score_reference_tokens(
            &prompt,
            &vision,
            NUMERICAL_PARITY_TOKENS,
            &permit,
            &cancellation,
        )
        .unwrap();
    let generated = session
        .decoder
        .generate(
            &prompt,
            &vision,
            NUMERICAL_PARITY_TOKENS.len(),
            &permit,
            &cancellation,
        )
        .unwrap();
    let generated_raw = session.tokenizer.decode(&generated).unwrap();
    eprintln!("Unlimited-OCR free-running tokens: {generated:?}");
    eprintln!("Unlimited-OCR free-running text: {generated_raw}");
    let finished = started.elapsed();

    for (step, score) in scores.iter().enumerate() {
        eprintln!(
            "Unlimited-OCR reference step={step} expected={} greedy={} rank={} expected_logit={} max_logit={} delta={}",
            score.expected_token,
            score.greedy_token,
            score.rank,
            score.expected_logit,
            score.max_logit,
            score.max_logit - score.expected_logit,
        );
    }
    let first_unstable = scores
        .iter()
        .position(|score| score.greedy_token != score.expected_token);
    eprintln!("Unlimited-OCR first unstable reference step: {first_unstable:?}");
    let unstable = scores
        .iter()
        .filter(|score| score.greedy_token != score.expected_token)
        .count();
    let selected = scores
        .iter()
        .map(|score| score.greedy_token)
        .collect::<Vec<_>>();
    assert_eq!(
        &selected[..NUMERICAL_PARITY_EXACT_PREFIX],
        &NUMERICAL_PARITY_TOKENS[..NUMERICAL_PARITY_EXACT_PREFIX],
        "the reviewed exact-token prefix regressed"
    );
    assert!(
        unstable <= NUMERICAL_PARITY_MAX_UNSTABLE_STEPS,
        "{unstable} teacher-forced steps diverged from the pinned upstream CPU reference; first unstable step: {first_unstable:?}"
    );
    for (step, score) in scores.iter().enumerate() {
        assert!(
            score.rank <= NUMERICAL_PARITY_MAX_EXPECTED_RANK,
            "the upstream token rank {} at step {step} exceeded {NUMERICAL_PARITY_MAX_EXPECTED_RANK}",
            score.rank,
        );
        assert!(
            score.max_logit - score.expected_logit <= NUMERICAL_PARITY_MAX_LOGIT_DELTA,
            "the upstream token logit delta at step {step} exceeded {NUMERICAL_PARITY_MAX_LOGIT_DELTA}",
        );
    }
    if std::env::var_os("A3S_UNLIMITED_OCR_REQUIRE_EXACT_PARITY").is_some() {
        assert_eq!(
            selected, NUMERICAL_PARITY_TOKENS,
            "strict teacher-forced token parity failed; first unstable step: {first_unstable:?}"
        );
    }

    let raw = session.tokenizer.decode(NUMERICAL_PARITY_TOKENS).unwrap();
    assert_eq!(raw, NUMERICAL_PARITY_TEXT);
    let geometry = source_grounding_geometry(&bytes).unwrap();
    let parsed = parse_model_output(&raw, geometry).unwrap();
    assert_eq!(parsed.blocks.len(), 3);
    assert_eq!(parsed.blocks[0].text, ".com 中国收藏热线");
    assert_eq!(
        parsed.blocks[0].category.as_ref().unwrap().raw_label,
        "header"
    );
    assert_eq!(
        parsed.blocks[0].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 24,
            y: 0,
            width: 172,
            height: 35,
        })
    );
    assert_eq!(
        parsed.blocks[1].category.as_ref().unwrap().raw_label,
        "title"
    );
    assert_eq!(parsed.blocks[1].text, "登机牌 BOARDING PA");
    assert_eq!(
        parsed.blocks[1].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 24,
            y: 25,
            width: 616,
            height: 51,
        })
    );
    assert_eq!(parsed.blocks[2].text, "FLIGHT 日期 DATE");
    assert_eq!(
        parsed.blocks[2].category.as_ref().unwrap().raw_label,
        "text"
    );
    assert_eq!(
        parsed.blocks[2].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 0,
            y: 103,
            width: 640,
            height: 26,
        })
    );

    let generated_parsed = parse_model_output(&generated_raw, geometry).unwrap();
    assert!(generated_parsed.warnings.is_empty());
    assert_eq!(generated_parsed.blocks.len(), parsed.blocks.len());
    assert!(matches!(
        generated_parsed.blocks[0].text.as_str(),
        ".com 中国收藏热线" | "com 中国收藏热线"
    ));
    assert_eq!(generated_parsed.blocks[1].text, parsed.blocks[1].text);
    assert_eq!(generated_parsed.blocks[2].text, parsed.blocks[2].text);
    for (actual, expected) in generated_parsed.blocks.iter().zip(&parsed.blocks) {
        assert_eq!(actual.category, expected.category);
        assert_bounding_box_within(actual.bounding_box, expected.bounding_box, 3);
        assert_eq!(actual.bounding_boxes.len(), expected.bounding_boxes.len());
        for (actual, expected) in actual.bounding_boxes.iter().zip(&expected.bounding_boxes) {
            assert_bounding_box_within(Some(*actual), Some(*expected), 3);
        }
    }

    eprintln!(
        "Unlimited-OCR parity device={} load={loaded:?} preprocess={:?} vision={:?} decode={:?} total={finished:?}",
        session.runtime.device().name(),
        preprocessed - loaded,
        encoded - preprocessed,
        finished - encoded,
    );
}

fn assert_bounding_box_within(
    actual: Option<crate::OcrBoundingBox>,
    expected: Option<crate::OcrBoundingBox>,
    tolerance: u32,
) {
    let actual = actual.expect("the local model must emit a reviewed source box");
    let expected = expected.expect("the upstream reference must contain a source box");
    assert!(actual.x.abs_diff(expected.x) <= tolerance);
    assert!(actual.y.abs_diff(expected.y) <= tolerance);
    assert!(actual.width.abs_diff(expected.width) <= tolerance);
    assert!(actual.height.abs_diff(expected.height) <= tolerance);
}

fn numerical_parity_crop(source: &[u8]) -> Vec<u8> {
    let image = image::load_from_memory(source)
        .expect("the pinned parity source image must decode")
        .to_rgb8();
    assert_eq!(image.dimensions(), (896, 528));
    let crop = image::imageops::crop_imm(&image, 128, 0, 640, 528).to_image();
    assert_eq!(
        format!("{:x}", Sha256::digest(crop.as_raw())),
        NUMERICAL_PARITY_CROP_RGB_SHA256
    );
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(crop.as_raw(), 640, 528, image::ExtendedColorType::Rgb8)
        .expect("the reviewed parity crop must encode losslessly");
    bytes
}

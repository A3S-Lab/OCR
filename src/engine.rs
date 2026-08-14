use a3s_power::inference::ExecutionReceipt;
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use imageproc::point::Point;
use tokio_util::sync::CancellationToken;

use crate::assets::ModelAssets;
use crate::cancellation::check_cancelled;
use crate::config::{load_detection, load_recognition, DetectionConfig, RecognitionConfig};
use crate::postprocess::detection_boxes_in_content;
use crate::ppocr_v6::native::NativePpOcrV6;
use crate::preprocess::detection_batch_inputs;

mod recognition;

#[derive(Debug, Clone)]
pub(crate) struct EngineBlock {
    pub(crate) polygon: [Point<f32>; 4],
    pub(crate) detection_confidence: f32,
    pub(crate) text: String,
    pub(crate) confidence: f32,
}

pub(crate) struct EngineExtraction {
    pub(crate) blocks: Vec<EngineBlock>,
    pub(crate) receipts: Vec<ExecutionReceipt>,
}

pub(crate) struct PpOcrV6Engine {
    native: NativePpOcrV6,
    detection_config: DetectionConfig,
    recognition_config: RecognitionConfig,
}

impl PpOcrV6Engine {
    #[cfg(test)]
    pub(crate) fn load(assets: &ModelAssets) -> UseResult<Self> {
        let detection_config = load_detection(&assets.detection_config)?;
        let recognition_config = load_recognition(&assets.recognition_config)?;
        let native = NativePpOcrV6::load(assets)?;
        Ok(Self {
            native,
            detection_config,
            recognition_config,
        })
    }

    pub(crate) fn load_with_runtime(
        assets: &ModelAssets,
        runtime: a3s_power::inference::EmbeddedRuntime,
    ) -> UseResult<Self> {
        let detection_config = load_detection(&assets.detection_config)?;
        let recognition_config = load_recognition(&assets.recognition_config)?;
        let native = NativePpOcrV6::load_with_runtime(assets, runtime)?;
        Ok(Self {
            native,
            detection_config,
            recognition_config,
        })
    }

    #[cfg(test)]
    pub(crate) fn extract(
        &mut self,
        image: &RgbImage,
        cancellation: &CancellationToken,
    ) -> UseResult<EngineExtraction> {
        check_cancelled(cancellation)?;
        let permit = self.native.begin(cancellation)?;
        self.extract_admitted(image, &permit, cancellation)
    }

    #[cfg(test)]
    pub(crate) fn extract_admitted(
        &mut self,
        image: &RgbImage,
        permit: &a3s_power::inference::ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<EngineExtraction> {
        self.extract_batch_admitted(&[image], permit, cancellation)?
            .pop()
            .ok_or_else(|| {
                engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 returned no output for a single admitted image.",
                )
            })?
    }

    pub(crate) fn extract_batch_admitted(
        &mut self,
        images: &[&RgbImage],
        permit: &a3s_power::inference::ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<UseResult<EngineExtraction>>> {
        check_cancelled(cancellation)?;
        let mut inputs = detection_batch_inputs(images, &self.detection_config)?;
        let graph_inputs = inputs
            .iter_mut()
            .map(|input| (std::mem::take(&mut input.data), input.shape))
            .collect();
        let detection = self
            .native
            .detect_batch(graph_inputs, permit, cancellation)?;
        if detection.tensors.len() != images.len() || inputs.len() != images.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection changed exact batch cardinality.",
            ));
        }
        let mut detections = Vec::with_capacity(images.len());
        for (input, tensor) in inputs.into_iter().zip(detection.tensors) {
            check_cancelled(cancellation)?;
            detections.push(detection_boxes_in_content(
                &tensor.values,
                &tensor.shape,
                input.content_width,
                input.content_height,
                input.original_width,
                input.original_height,
                &self.detection_config,
            ));
        }
        self.recognize_detected_batch(images, detections, detection.receipt, permit, cancellation)
    }
}

fn engine_error(code: &str, message: impl Into<String>) -> UseError {
    UseError::new(code, message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::assets::OcrInstallSource;
    use crate::postprocess::Detection;

    const REAL_IMAGE_SHA256: &str =
        "4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4";
    const MAX_RECOGNITION_SCORE_DELTA: f64 = 0.065;
    const MAX_BATCH_RECOGNITION_SCORE_DELTA: f32 = 0.000_01;
    const MAX_POLYGON_COORDINATE_DELTA: f32 = 4.0;

    struct ExpectedBlock {
        text: &'static str,
        recognition: f64,
        polygon: [(i32, i32); 4],
    }

    // Generated once with Paddle 3.3.1, PaddleOCR 3.7.0, and the exact
    // PP-OCRv6 small artifacts pinned by this crate. The gate below executes
    // only the Rust implementation; Paddle and Python are not test or runtime
    // dependencies.
    const UPSTREAM_BLOCKS: &[ExpectedBlock] = &[
        ExpectedBlock {
            text: "www.997788.com中国收藏热线",
            recognition: 0.998620331,
            polygon: [(0, 1), (335, 0), (335, 33), (0, 34)],
        },
        ExpectedBlock {
            text: "登机牌",
            recognition: 0.999989033,
            polygon: [(150, 22), (358, 15), (360, 74), (152, 81)],
        },
        ExpectedBlock {
            text: "BOARDING",
            recognition: 0.999916852,
            polygon: [(418, 19), (661, 15), (661, 61), (418, 64)],
        },
        ExpectedBlock {
            text: "PASS",
            recognition: 0.999979794,
            polygon: [(699, 13), (823, 10), (824, 60), (700, 62)],
        },
        ExpectedBlock {
            text: "舱位 CLASS",
            recognition: 0.953512967,
            polygon: [(340, 103), (458, 103), (458, 128), (340, 128)],
        },
        ExpectedBlock {
            text: "序号SERIAL NO.",
            recognition: 0.993820965,
            polygon: [(486, 100), (649, 98), (649, 123), (486, 125)],
        },
        ExpectedBlock {
            text: "座位号 SEAT NO.",
            recognition: 0.980658054,
            polygon: [(674, 95), (835, 91), (836, 118), (675, 123)],
        },
        ExpectedBlock {
            text: "航班FLIGHT",
            recognition: 0.999896109,
            polygon: [(63, 110), (192, 108), (192, 130), (63, 132)],
        },
        ExpectedBlock {
            text: "日期 DATE",
            recognition: 0.983355403,
            polygon: [(212, 106), (318, 106), (318, 131), (212, 131)],
        },
        ExpectedBlock {
            text: "MU 2379",
            recognition: 0.987337768,
            polygon: [(81, 138), (214, 136), (214, 161), (81, 163)],
        },
        ExpectedBlock {
            text: "03DEC",
            recognition: 0.984453499,
            polygon: [(231, 136), (327, 134), (327, 160), (231, 162)],
        },
        ExpectedBlock {
            text: "W",
            recognition: 0.991825104,
            polygon: [(405, 133), (430, 133), (430, 157), (405, 157)],
        },
        ExpectedBlock {
            text: "035",
            recognition: 0.999922514,
            polygon: [(509, 129), (568, 129), (568, 156), (509, 156)],
        },
        ExpectedBlock {
            text: "始发地 FROM",
            recognition: 0.948752105,
            polygon: [(341, 172), (470, 169), (470, 195), (341, 198)],
        },
        ExpectedBlock {
            text: "登机口",
            recognition: 0.999819696,
            polygon: [(487, 173), (553, 171), (554, 194), (488, 196)],
        },
        ExpectedBlock {
            text: "GATE",
            recognition: 0.999981642,
            polygon: [(565, 171), (615, 171), (615, 194), (565, 194)],
        },
        ExpectedBlock {
            text: "登机时间BDT",
            recognition: 0.999968469,
            polygon: [(676, 167), (811, 164), (811, 190), (676, 193)],
        },
        ExpectedBlock {
            text: "目的地 TO",
            recognition: 0.906847239,
            polygon: [(66, 178), (169, 178), (169, 203), (66, 203)],
        },
        ExpectedBlock {
            text: "福州",
            recognition: 0.999795079,
            polygon: [(97, 206), (168, 206), (168, 229), (97, 229)],
        },
        ExpectedBlock {
            text: "TAIYUAN",
            recognition: 0.999807537,
            polygon: [(336, 216), (476, 216), (476, 240), (336, 240)],
        },
        ExpectedBlock {
            text: "G11",
            recognition: 0.999913216,
            polygon: [(506, 213), (553, 213), (553, 237), (506, 237)],
        },
        ExpectedBlock {
            text: "FUZHOU",
            recognition: 0.999510050,
            polygon: [(88, 226), (204, 226), (204, 252), (88, 252)],
        },
        ExpectedBlock {
            text: "身份识别ID NO.",
            recognition: 0.995628238,
            polygon: [(342, 236), (485, 233), (485, 258), (342, 261)],
        },
        ExpectedBlock {
            text: "姓名 NAME",
            recognition: 0.944117248,
            polygon: [(65, 250), (172, 247), (172, 269), (65, 271)],
        },
        ExpectedBlock {
            text: "ZHANGQIWEI",
            recognition: 0.999886990,
            polygon: [(74, 274), (264, 272), (264, 298), (74, 300)],
        },
        ExpectedBlock {
            text: "票号 TKT NO.",
            recognition: 0.992853165,
            polygon: [(460, 294), (580, 292), (580, 317), (460, 320)],
        },
        ExpectedBlock {
            text: "张祺伟",
            recognition: 0.998239696,
            polygon: [(103, 311), (210, 311), (210, 337), (103, 337)],
        },
        ExpectedBlock {
            text: "票价 FARE",
            recognition: 0.984983921,
            polygon: [(68, 342), (166, 339), (166, 365), (68, 367)],
        },
        ExpectedBlock {
            text: "ETKT7813699238489/1",
            recognition: 0.999883354,
            polygon: [(343, 348), (663, 344), (663, 368), (343, 371)],
        },
        ExpectedBlock {
            text: "登机口于起飞前10分钟关闭 GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME",
            recognition: 0.978114545,
            polygon: [(98, 455), (832, 441), (832, 466), (98, 480)],
        },
    ];

    #[test]
    #[ignore = "requires the pinned official PP-OCRv6 bundle and real-image fixture"]
    fn official_real_image_matches_upstream() {
        let root = PathBuf::from(
            std::env::var_os("A3S_PPOCR_V6_MODEL")
                .expect("A3S_PPOCR_V6_MODEL must name the pinned official model bundle"),
        );
        let image_path = PathBuf::from(
            std::env::var_os("A3S_PPOCR_V6_REAL_IMAGE")
                .expect("A3S_PPOCR_V6_REAL_IMAGE must name the pinned official image"),
        );
        let assets = ModelAssets {
            root: root.clone(),
            detection_weights: root.join("det/model.safetensors"),
            detection_config: root.join("det/inference.yml"),
            recognition_weights: root.join("rec/model.safetensors"),
            recognition_config: root.join("rec/inference.yml"),
            source: OcrInstallSource::Environment,
        };
        let mut engine = PpOcrV6Engine::load(&assets).unwrap();
        let bytes = std::fs::read(image_path).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), REAL_IMAGE_SHA256);
        let image = crate::preprocess::decode_image(&bytes).unwrap();
        assert_eq!(image.dimensions(), (896, 528));
        let cancellation = CancellationToken::new();
        let extraction = engine.extract(&image, &cancellation).unwrap();
        assert_eq!(extraction.receipts.len(), 5);
        assert_eq!(extraction.blocks.len(), UPSTREAM_BLOCKS.len());

        for (index, (actual, expected)) in extraction.blocks.iter().zip(UPSTREAM_BLOCKS).enumerate()
        {
            assert_eq!(
                parity_text(&actual.text),
                parity_text(expected.text),
                "recognized text diverged at block {index}: {:?} versus {:?}",
                actual.text,
                expected.text
            );
            assert!(
                (f64::from(actual.confidence) - expected.recognition).abs()
                    <= MAX_RECOGNITION_SCORE_DELTA,
                "recognition score diverged at block {index}: {} versus {}",
                actual.confidence,
                expected.recognition
            );
            assert!(actual.detection_confidence >= engine.detection_config.box_threshold);
            for (point_index, (actual, expected)) in
                actual.polygon.iter().zip(expected.polygon).enumerate()
            {
                let x_delta = (actual.x - expected.0 as f32).abs();
                let y_delta = (actual.y - expected.1 as f32).abs();
                assert!(
                    x_delta <= MAX_POLYGON_COORDINATE_DELTA
                        && y_delta <= MAX_POLYGON_COORDINATE_DELTA,
                    "polygon diverged at block {index}, point {point_index}: {actual:?} versus {expected:?}"
                );
            }
        }

        let first_block = &extraction.blocks[0];
        let detection = Detection {
            polygon: first_block.polygon,
            confidence: first_block.detection_confidence,
        };
        let scalar_permit = engine.native.begin(&cancellation).unwrap();
        let scalar = engine
            .recognize_detected_batch(
                &[&image],
                vec![Ok(vec![detection.clone()])],
                extraction.receipts[0].clone(),
                &scalar_permit,
                &cancellation,
            )
            .unwrap();
        let scalar = scalar[0].as_ref().unwrap();
        drop(scalar_permit);
        let batch_permit = engine.native.begin(&cancellation).unwrap();
        let batch = engine
            .recognize_detected_batch(
                &[&image, &image],
                vec![Ok(vec![detection.clone()]), Ok(vec![detection])],
                extraction.receipts[0].clone(),
                &batch_permit,
                &cancellation,
            )
            .unwrap();
        let first = batch[0].as_ref().unwrap();
        let second = batch[1].as_ref().unwrap();
        for batched in [first, second] {
            assert_eq!(batched.blocks.len(), 1);
            assert_eq!(batched.blocks[0].text, scalar.blocks[0].text);
            assert!(
                (batched.blocks[0].confidence - scalar.blocks[0].confidence).abs()
                    <= MAX_BATCH_RECOGNITION_SCORE_DELTA
            );
            assert_eq!(batched.blocks[0].polygon, scalar.blocks[0].polygon);
            assert_eq!(batched.receipts.len(), 2);
        }
        let scalar_recognition = &scalar.receipts[1];
        let shared_recognition = &first.receipts[1];
        assert_eq!(shared_recognition, &second.receipts[1]);
        assert!(shared_recognition.model.family.ends_with("-recognition"));
        assert_eq!(
            shared_recognition.input.byte_length,
            scalar_recognition.input.byte_length * 2
        );
        assert_eq!(
            shared_recognition.input.item_count,
            scalar_recognition.input.item_count * 2
        );
    }

    fn parity_text(value: &str) -> String {
        value
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>()
            .replace(".中国", "中国")
    }
}

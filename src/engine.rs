use a3s_power::inference::ExecutionReceipt;
use a3s_use_core::{UseError, UseResult};
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use imageproc::point::Point;
use tokio_util::sync::CancellationToken;

use crate::assets::ModelAssets;
use crate::cancellation::check_cancelled;
use crate::config::{load_detection, load_recognition, DetectionConfig, RecognitionConfig};
use crate::postprocess::{decode_ctc, detection_boxes_in_content, Detection};
use crate::ppocr_v6::native::NativePpOcrV6;
use crate::preprocess::{detection_batch_inputs, recognition_input};

const RECOGNITION_BATCH_SIZE: usize = 8;
const MAX_CROP_PIXELS: u64 = 64 * 1024 * 1024;

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
        let mut outputs = Vec::with_capacity(images.len());
        for ((image, input), tensor) in images.iter().zip(inputs).zip(detection.tensors) {
            check_cancelled(cancellation)?;
            let detections = detection_boxes_in_content(
                &tensor.values,
                &tensor.shape,
                input.content_width,
                input.content_height,
                input.original_width,
                input.original_height,
                &self.detection_config,
            );
            outputs.push(detections.and_then(|detections| {
                self.recognize_detections(
                    image,
                    detections,
                    detection.receipt.clone(),
                    permit,
                    cancellation,
                )
            }));
        }
        Ok(outputs)
    }

    fn recognize_detections(
        &mut self,
        image: &RgbImage,
        detections: Vec<Detection>,
        detection_receipt: ExecutionReceipt,
        permit: &a3s_power::inference::ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<EngineExtraction> {
        if detections.is_empty() {
            return Ok(EngineExtraction {
                blocks: Vec::new(),
                receipts: vec![detection_receipt],
            });
        }

        let crops = detections
            .iter()
            .map(|detection| perspective_crop(image, detection))
            .collect::<UseResult<Vec<_>>>()?;
        let mut blocks = Vec::with_capacity(detections.len());
        let mut receipts = vec![detection_receipt];
        for (detection_batch, crop_batch) in detections
            .chunks(RECOGNITION_BATCH_SIZE)
            .zip(crops.chunks(RECOGNITION_BATCH_SIZE))
        {
            check_cancelled(cancellation)?;
            let input = recognition_input(crop_batch, &self.recognition_config)?;
            let recognition =
                self.native
                    .recognize(input.data, input.shape, permit, cancellation)?;
            let shape = recognition.tensor.shape;
            let output = recognition.tensor.values;
            receipts.push(recognition.receipt);
            if shape.len() != 3 || shape[0] != detection_batch.len() {
                return Err(engine_error(
                    "use.ocr.provider_output_invalid",
                    format!(
                        "PP-OCRv6 recognition output shape must be [N, T, C] for N={}, found {shape:?}.",
                        detection_batch.len()
                    ),
                ));
            }
            let item_len = shape[1].checked_mul(shape[2]).ok_or_else(|| {
                engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output dimensions overflowed.",
                )
            })?;
            if output.len() != detection_batch.len().saturating_mul(item_len) {
                return Err(engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output length does not match its batch shape.",
                ));
            }
            for (index, detection) in detection_batch.iter().enumerate() {
                let start = index * item_len;
                let recognition = decode_ctc(
                    &output[start..start + item_len],
                    &[1, shape[1], shape[2]],
                    &self.recognition_config,
                )?;
                blocks.push(EngineBlock {
                    polygon: detection.polygon,
                    detection_confidence: detection.confidence,
                    text: recognition.text,
                    confidence: recognition.confidence,
                });
            }
        }
        Ok(EngineExtraction { blocks, receipts })
    }
}

fn perspective_crop(image: &RgbImage, detection: &Detection) -> UseResult<RgbImage> {
    let polygon = detection.polygon;
    let width = distance(polygon[0], polygon[1])
        .max(distance(polygon[2], polygon[3]))
        .max(1.0) as u32;
    let height = distance(polygon[0], polygon[3])
        .max(distance(polygon[1], polygon[2]))
        .max(1.0) as u32;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| crop_error("PP-OCRv6 text crop dimensions overflowed."))?;
    if pixels > MAX_CROP_PIXELS {
        return Err(crop_error(
            "PP-OCRv6 text crop exceeds the 64 megapixel safety limit.",
        ));
    }

    let source = polygon.map(|point| (point.x, point.y));
    let destination = [
        (0.0, 0.0),
        (width as f32, 0.0),
        (width as f32, height as f32),
        (0.0, height as f32),
    ];
    let projection = Projection::from_control_points(source, destination).ok_or_else(|| {
        crop_error("PP-OCRv6 detected a degenerate text polygon that cannot be rectified.")
    })?;
    let mut crop = ImageBuffer::new(width, height);
    warp_into(
        image,
        &projection,
        Interpolation::Bicubic,
        Rgb([255, 255, 255]),
        &mut crop,
    );
    if f64::from(height) / f64::from(width) >= 1.5 {
        Ok(imageops::rotate270(&crop))
    } else {
        Ok(crop)
    }
}

fn distance(left: Point<f32>, right: Point<f32>) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn crop_error(message: impl Into<String>) -> UseError {
    engine_error("use.ocr.crop_invalid", message)
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

    const REAL_IMAGE_SHA256: &str =
        "4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4";
    const MAX_RECOGNITION_SCORE_DELTA: f64 = 0.065;
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
    fn perspective_crop_uses_the_upstream_exclusive_extent() {
        let image = RgbImage::new(20, 20);
        let detection = Detection {
            polygon: [
                Point::new(1.0, 2.0),
                Point::new(11.0, 2.0),
                Point::new(11.0, 7.0),
                Point::new(1.0, 7.0),
            ],
            confidence: 1.0,
        };
        let crop = perspective_crop(&image, &detection).unwrap();
        assert_eq!(crop.dimensions(), (10, 5));
    }

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
    }

    fn parity_text(value: &str) -> String {
        value
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>()
            .replace(".中国", "中国")
    }
}

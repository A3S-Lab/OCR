use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use a3s_use_core::Artifact;
use image::ImageEncoder as _;

use super::grounding::{GroundingGeometry, MAX_GROUNDING_BLOCK_TEXT_BYTES};
use super::*;

#[test]
fn local_and_remote_endpoints_preserve_the_data_boundary() {
    let local = UnlimitedOcrConfig::local("http://127.0.0.1:8000").unwrap();
    assert_eq!(local.base_url(), "http://127.0.0.1:8000/v1/");
    assert!(!local.sends_source_off_device());
    assert!(UnlimitedOcrConfig::local("http://[::1]:8000/v1").is_ok());

    let remote = UnlimitedOcrConfig::remote("https://ocr.example.com/v1").unwrap();
    assert_eq!(remote.base_url(), "https://ocr.example.com/v1/");
    assert!(remote.sends_source_off_device());
    assert!(UnlimitedOcrConfig::local("http://ocr.example.com/v1").is_err());
    assert!(UnlimitedOcrConfig::remote("http://ocr.example.com/v1").is_err());
}

#[test]
fn debug_output_redacts_bearer_tokens() {
    let config = UnlimitedOcrConfig::local("http://localhost:8000/v1")
        .unwrap()
        .with_bearer_token("very-secret-token")
        .unwrap();
    let debug = format!("{config:?}");
    assert!(debug.contains("[REDACTED]"));
    assert!(!debug.contains("very-secret-token"));
}

#[test]
fn parses_reviewed_grounding_forms_into_source_pixel_blocks() {
    let raw = concat!(
        "<|ref|>title<|/ref|><|det|>[[0, 0, 999, 100]]<|/det|>",
        "# Heading\ncontinued\n",
        "<|det|>text [[100, 200, 400, 300], [500, 250, 900, 350]]<|/det|>",
        "Body \\coloneqq value",
        "<｜end▁of▁sentence｜>"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 1_000,
            height: 500,
        }),
    )
    .unwrap();

    assert_eq!(parsed.text, "# Heading\ncontinued\nBody := value");
    assert!(parsed.warnings.is_empty());
    assert_eq!(parsed.blocks.len(), 2);
    assert_eq!(parsed.blocks[0].page, 1);
    assert_eq!(parsed.blocks[0].text, "# Heading\ncontinued");
    assert_eq!(
        parsed.blocks[0]
            .category
            .as_ref()
            .map(|category| (category.raw_label.as_str(), category.role,)),
        Some(("title", crate::OcrBlockRole::Title))
    );
    assert_eq!(
        parsed.blocks[0].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 0,
            y: 0,
            width: 1_000,
            height: 50,
        })
    );
    assert_eq!(
        parsed.blocks[0].bounding_boxes,
        [crate::OcrBoundingBox {
            x: 0,
            y: 0,
            width: 1_000,
            height: 50,
        }]
    );
    assert_eq!(parsed.blocks[1].text, "Body := value");
    assert_eq!(
        parsed.blocks[1]
            .category
            .as_ref()
            .map(|category| (category.raw_label.as_str(), category.role,)),
        Some(("text", crate::OcrBlockRole::Text))
    );
    assert_eq!(
        parsed.blocks[1].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 100,
            y: 100,
            width: 800,
            height: 75,
        })
    );
    assert_eq!(
        parsed.blocks[1].bounding_boxes,
        [
            crate::OcrBoundingBox {
                x: 100,
                y: 100,
                width: 300,
                height: 50,
            },
            crate::OcrBoundingBox {
                x: 500,
                y: 125,
                width: 400,
                height: 50,
            },
        ]
    );
    assert!(parsed.blocks.iter().all(|block| block.confidence.is_none()
        && block.detection_confidence.is_none()
        && block.polygon.is_none()));
}

#[test]
fn preserves_open_labels_with_conservative_canonical_roles() {
    let raw = concat!(
        "<|det|>equation_isolated [0, 0, 100, 100]<|/det|>E = mc^2\n",
        "<|det|>figure_caption [0, 200, 100, 300]<|/det|>Figure 1\n",
        "<|det|>vendor-special [0, 400, 100, 500]<|/det|>Opaque"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 999,
            height: 999,
        }),
    )
    .unwrap();

    assert!(parsed.warnings.is_empty());
    assert_eq!(parsed.blocks.len(), 3);
    assert_eq!(
        parsed
            .blocks
            .iter()
            .map(|block| {
                let category = block.category.as_ref().unwrap();
                (category.raw_label.as_str(), category.role)
            })
            .collect::<Vec<_>>(),
        [
            ("equation_isolated", crate::OcrBlockRole::EquationBlock),
            ("figure_caption", crate::OcrBlockRole::Caption),
            ("vendor-special", crate::OcrBlockRole::Unknown),
        ]
    );
}

#[test]
fn preserves_unclosed_grounding_markers() {
    let raw = "text <|det|>unfinished";
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();
    assert_eq!(parsed.text, raw);
    assert!(parsed.blocks.is_empty());
    assert_eq!(parsed.warnings.len(), 1);
}

#[test]
fn malformed_and_non_text_grounding_never_fabricate_boxes() {
    let raw = concat!(
        "<|det|>image [0, 0, 999, 999]<|/det|>![figure](figure.jpg)\n",
        "<|det|>text [0, 0, 1000, 10]<|/det|>Out of range\n",
        "<|det|>text [10, 20, 10, 30]<|/det|>Collapsed\n",
        "Plain text"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();

    assert_eq!(
        parsed.text,
        "![figure](figure.jpg)\nOut of range\nCollapsed\nPlain text"
    );
    assert!(parsed.blocks.is_empty());
    assert_eq!(parsed.warnings.len(), 1);
}

#[test]
fn grounding_marker_count_is_bounded_before_block_construction() {
    let raw = "<|det|>text [0, 0, 1, 1]<|/det|>x".repeat(MAX_GROUNDING_MARKERS + 1);
    let error = parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap_err();

    assert_eq!(error.code, "use.ocr.provider_output_invalid");
}

#[test]
fn grounding_coordinate_count_and_source_dimensions_are_bounded() {
    let boxes = (0..129)
        .map(|_| "[0, 0, 1, 1]")
        .collect::<Vec<_>>()
        .join(",");
    let raw = format!("<|det|>text [{boxes}]<|/det|>bounded");
    let error = parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, "use.ocr.provider_output_invalid");

    let error = source_grounding_geometry(b"\x89PNG\r\n\x1a\ninvalid").unwrap_err();
    assert_eq!(error.code, "use.ocr.provider_output_invalid");
}

#[test]
fn oversized_grounded_text_is_preserved_without_geometry() {
    let text = "x".repeat(MAX_GROUNDING_BLOCK_TEXT_BYTES + 1);
    let raw = format!("<|det|>text [0, 0, 999, 999]<|/det|>{text}");
    let parsed = parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();

    assert_eq!(parsed.text, text);
    assert!(parsed.blocks.is_empty());
    assert_eq!(parsed.warnings.len(), 1);
}

#[test]
fn missing_or_orientation_transformed_grounding_is_explicitly_degraded() {
    let plain = parse_model_output(
        "plain text",
        Some(GroundingGeometry {
            width: 3,
            height: 2,
        }),
    )
    .unwrap();
    assert!(plain.blocks.is_empty());
    assert_eq!(plain.warnings.len(), 1);

    let image_only = parse_model_output(
        "<|det|>image [0, 0, 999, 999]<|/det|>![figure](figure.jpg)",
        Some(GroundingGeometry {
            width: 3,
            height: 2,
        }),
    )
    .unwrap();
    assert!(image_only.blocks.is_empty());
    assert_eq!(image_only.warnings.len(), 1);

    let raw = "<|det|>text [0, 0, 999, 999]<|/det|>grounded";
    let transformed = parse_model_output(raw, None).unwrap();
    assert_eq!(transformed.text, "grounded");
    assert!(transformed.blocks.is_empty());
    assert_eq!(transformed.warnings.len(), 1);

    let oriented_jpeg = fixture_oriented_jpeg(3, 2);
    assert_eq!(source_grounding_geometry(&oriented_jpeg).unwrap(), None);
}

#[tokio::test]
async fn sends_the_official_vllm_request_and_decodes_its_response() {
    let response_body = serde_json::json!({
        "choices": [{
            "message": {
                "content": concat!(
                    "<|ref|>title<|/ref|>",
                    "<|det|>[[10, 20, 900, 80]]<|/det|>",
                    "# Parsed heading",
                    "<｜end▁of▁sentence｜>"
                )
            }
        }]
    })
    .to_string();
    let (base_url, request_thread) = serve_once(response_body);
    let provider = UnlimitedOcrProvider::new(UnlimitedOcrConfig::local(base_url).unwrap()).unwrap();
    let image = fixture_png(100, 50);
    let input = OcrInput::new(
        Artifact {
            path: "/tmp/fixture.png".into(),
            media_type: "image/png".to_string(),
            size: image.len() as u64,
            sha256: "fixture-digest".to_string(),
        },
        image.clone(),
    );

    let output = provider.recognize(input).await.unwrap();

    assert_eq!(output.model.as_deref(), Some(UNLIMITED_OCR_MODEL));
    assert_eq!(output.text, "# Parsed heading");
    assert_eq!(output.blocks.len(), 1);
    assert_eq!(output.blocks[0].text, "# Parsed heading");
    assert_eq!(
        output.blocks[0].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 1,
            y: 1,
            width: 89,
            height: 3,
        })
    );
    assert!(output.warnings.is_empty());

    let request = request_thread.join().unwrap();
    let (head, body) = request.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    let payload: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(payload["model"], UNLIMITED_OCR_MODEL);
    assert_eq!(payload["messages"][0]["content"][0]["text"], PROMPT);
    assert_eq!(payload["skip_special_tokens"], false);
    assert_eq!(payload["vllm_xargs"]["ngram_size"], 35);
    assert_eq!(payload["vllm_xargs"]["window_size"], 128);
    assert_eq!(
        payload["messages"][0]["content"][1]["image_url"]["url"],
        format!("data:image/png;base64,{}", BASE64_STANDARD.encode(image))
    );
}

fn fixture_png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            &vec![255; (width * height) as usize],
            width,
            height,
            image::ExtendedColorType::L8,
        )
        .unwrap();
    bytes
}

fn fixture_oriented_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
        .encode(
            &vec![255; (width * height * 3) as usize],
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    let exif = [
        b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0,
        0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];
    let segment_length = u16::try_from(exif.len() + 2).unwrap().to_be_bytes();
    let mut oriented = Vec::with_capacity(jpeg.len() + exif.len() + 4);
    oriented.extend_from_slice(&jpeg[..2]);
    oriented.extend_from_slice(&[0xff, 0xe1]);
    oriented.extend_from_slice(&segment_length);
    oriented.extend_from_slice(&exif);
    oriented.extend_from_slice(&jpeg[2..]);
    oriented
}

fn serve_once(response_body: String) -> (String, thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 8 * 1024];
        loop {
            let count = stream.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..count]);
            if request_is_complete(&request) {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).unwrap();
        String::from_utf8(request).unwrap()
    });
    (format!("http://{address}/v1"), handle)
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
        return false;
    };
    let header_end = header_end + 4;
    let Ok(headers) = std::str::from_utf8(&request[..header_end]) else {
        return false;
    };
    let content_length = headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<usize>().ok())
            .flatten()
    });
    content_length.is_some_and(|length| request.len() >= header_end + length)
}

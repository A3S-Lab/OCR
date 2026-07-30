use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use a3s_use_core::Artifact;

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
fn cleans_grounding_markers_without_rewriting_markdown() {
    let raw = concat!(
        "<|ref|>title<|/ref|><|det|>[[10, 20, 900, 80]]<|/det|># Heading\n\n",
        "<|det|>text [10, 100, 900, 200]<|/det|>Body \\coloneqq value",
        "<｜end▁of▁sentence｜>"
    );
    let (text, removed) = clean_model_output(raw);
    assert_eq!(text, "# Heading\n\nBody := value");
    assert!(removed);
}

#[test]
fn preserves_unclosed_grounding_markers() {
    let raw = "text <|det|>unfinished";
    let (text, removed) = clean_model_output(raw);
    assert_eq!(text, raw);
    assert!(!removed);
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
    let image = b"\x89PNG\r\n\x1a\nfixture".to_vec();
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
    assert!(output.blocks.is_empty());
    assert_eq!(output.warnings.len(), 1);

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

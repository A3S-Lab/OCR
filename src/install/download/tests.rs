use std::io::{Read as _, Write as _};
use std::net::{Shutdown, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::*;

fn response(
    status: &str,
    declared_length: usize,
    extra_headers: &[(&str, String)],
    body: &[u8],
) -> Vec<u8> {
    let mut value =
        format!("HTTP/1.1 {status}\r\nContent-Length: {declared_length}\r\nConnection: close\r\n");
    for (name, header_value) in extra_headers {
        value.push_str(&format!("{name}: {header_value}\r\n"));
    }
    value.push_str("\r\n");
    let mut value = value.into_bytes();
    value.extend_from_slice(body);
    value
}

fn scripted_server(
    responses: Vec<Vec<u8>>,
) -> (
    reqwest::Url,
    Arc<Mutex<Vec<String>>>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&requests);
    let server = thread::spawn(move || {
        for response in responses {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "timed out waiting for request");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("failed to accept request: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                assert_ne!(read, 0, "request ended before its headers");
                request.extend_from_slice(&buffer[..read]);
            }
            observed
                .lock()
                .unwrap()
                .push(String::from_utf8(request).unwrap());
            stream.write_all(&response).unwrap();
            stream.shutdown(Shutdown::Both).unwrap();
        }
    });
    (
        reqwest::Url::parse(&format!("http://{address}/bundle.tar")).unwrap(),
        requests,
        server,
    )
}

#[tokio::test]
async fn interrupted_download_resumes_from_the_verified_prefix() {
    let archive = b"immutable-model-archive";
    let split = 7;
    let content_range = format!("bytes {split}-{}/{}", archive.len() - 1, archive.len());
    let (url, requests, server) = scripted_server(vec![
        response("200 OK", archive.len(), &[], &archive[..split]),
        response(
            "206 Partial Content",
            archive.len() - split,
            &[("Content-Range", content_range)],
            &archive[split..],
        ),
    ]);
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("bundle.tar");
    let client = reqwest::Client::builder().build().unwrap();

    let downloaded = validated(
        &client,
        url,
        &destination,
        archive.len() as u64,
        Duration::ZERO,
    )
    .await
    .unwrap();

    server.join().unwrap();
    assert_eq!(std::fs::read(destination).unwrap(), archive);
    assert_eq!(downloaded.bytes, archive.len() as u64);
    assert_eq!(downloaded.sha256, format!("{:x}", Sha256::digest(archive)));
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(!requests[0].to_ascii_lowercase().contains("\r\nrange:"));
    assert!(requests[1]
        .to_ascii_lowercase()
        .contains(&format!("\r\nrange: bytes={split}-\r\n")));
}

#[tokio::test]
async fn range_ignoring_retry_restarts_without_appending_duplicate_bytes() {
    let archive = b"immutable-model-archive";
    let split = 5;
    let (url, requests, server) = scripted_server(vec![
        response("200 OK", archive.len(), &[], &archive[..split]),
        response("200 OK", archive.len(), &[], archive),
    ]);
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("bundle.tar");
    let client = reqwest::Client::builder().build().unwrap();

    validated(
        &client,
        url,
        &destination,
        archive.len() as u64,
        Duration::ZERO,
    )
    .await
    .unwrap();

    server.join().unwrap();
    assert_eq!(std::fs::read(destination).unwrap(), archive);
    let requests = requests.lock().unwrap();
    assert!(requests[1]
        .to_ascii_lowercase()
        .contains(&format!("\r\nrange: bytes={split}-\r\n")));
}

#[tokio::test]
async fn resumed_download_rejects_a_substituted_content_range() {
    let archive = b"immutable-model-archive";
    let split = 5;
    let substituted_range = format!("bytes 4-{}/{}", archive.len() - 1, archive.len());
    let (url, _requests, server) = scripted_server(vec![
        response("200 OK", archive.len(), &[], &archive[..split]),
        response(
            "206 Partial Content",
            archive.len() - split,
            &[("Content-Range", substituted_range)],
            &archive[split..],
        ),
    ]);
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("bundle.tar");
    let client = reqwest::Client::builder().build().unwrap();

    let result = validated(
        &client,
        url,
        &destination,
        archive.len() as u64,
        Duration::ZERO,
    )
    .await;

    server.join().unwrap();
    assert!(result.is_err());
    assert_eq!(std::fs::read(destination).unwrap(), &archive[..split]);
}

#[tokio::test]
async fn retryable_origin_failures_recover_after_a_short_outage() {
    let archive = b"immutable-model-archive";
    let mut responses = (0..5)
        .map(|_| response("503 Service Unavailable", 0, &[], &[]))
        .collect::<Vec<_>>();
    responses.push(response("200 OK", archive.len(), &[], archive));
    let (url, requests, server) = scripted_server(responses);
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("bundle.tar");
    let client = reqwest::Client::builder().build().unwrap();

    validated(
        &client,
        url,
        &destination,
        archive.len() as u64,
        Duration::ZERO,
    )
    .await
    .unwrap();

    server.join().unwrap();
    assert_eq!(std::fs::read(destination).unwrap(), archive);
    assert_eq!(requests.lock().unwrap().len(), 6);
}

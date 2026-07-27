# AGENTS.md

## Repository

This repository owns the typed `a3s-use-ocr` Rust library, standalone binary,
standard MCP server, packaged Skill, and pinned PP-OCRv6 model provenance.

## Boundaries

- Keep one OCR provider: local `PP-OCRv6_small` through ONNX Runtime.
- Do not add Python, PaddlePaddle, remote OCR APIs, or off-device fallbacks.
- Keep image inputs bounded and bind results to canonical source evidence.
- Preserve the `a3s-use-ocr` crate, binary, MCP, and Skill identities.
- A3S Use owns the built-in `ocr` route and product component policy.
- `a3s-use-core` owns shared machine contracts; depend on its released crate.

## Engineering

- Use Tokio for I/O and avoid blocking inside async contexts.
- Keep public types `Send + Sync` where applicable.
- Return typed contextual errors; avoid production panics.
- Keep all code and documentation in English.
- Run `cargo fmt --all`, focused tests, Clippy, and package verification before
  completion.

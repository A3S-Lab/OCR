# AGENTS.md

## Repository

This repository owns the provider-oriented `a3s-use-ocr` Rust library,
standalone binary, standard MCP server, packaged Skill, and the default
PP-OCRv6 provider.

## Boundaries

- Keep `OcrProvider` object-safe, `Send + Sync`, and independent of concrete
  provider dependencies.
- Inject providers as typed objects. Do not add raw backend-name switches or a
  global mutable provider registry.
- Keep source validation, bounds, media-type detection, and canonical evidence
  in `OcrClient`; providers must not replace source provenance.
- Keep PP-OCRv6 as one optional default provider. Its implementation remains
  local through ONNX Runtime and must not add Python or PaddlePaddle.
- Every provider descriptor must declare whether it sends source bytes off
  device; SDK and MCP surfaces must preserve that policy.
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

# A3S Use OCR

`a3s-use-ocr` is the independently maintained OCR engine used by A3S Use. The
repository publishes a typed Rust library, a standalone development CLI,
standard stdio MCP, the `a3s-use-ocr` Skill, and a content-bound release asset
bundle. A3S Use links the crate as its first-party built-in `ocr` route; moving
the source to this repository does not turn OCR into a generic extension.

A3S Code receives the release-matched capability as `mcp__use_ocr__*` without
installing a separate extension. The native CLI and standard MCP server share
one local PP-OCRv6 implementation.

There is one OCR provider:

- provider: `pp-ocr-v6`
- engine: `onnx-runtime`
- model bundle: `PP-OCRv6_small`

The release packages the pinned detection and recognition models. If the model
bundle is absent or damaged, install or repair it explicitly:

```bash
a3s install use/ocr
a3s install use/ocr --force
```

`A3S_OCR_MODEL_DIR` can point development builds at an explicit model bundle.
`A3S_USE_OCR_HOME` overrides the managed model root for packaging, tests, or an
isolated installation. Neither setting selects another OCR backend.

## Build

```bash
cargo fmt --all -- --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo build --release --locked
```

The library depends on the stable `a3s-use-core` machine contracts. A3S Use
pins an exact OCR release when assembling its built-in capability and release
assets.

## Workflow

For each bounded local image, the native engine:

1. decodes the image and applies PP-OCRv6 BGR normalization;
2. runs `PP-OCRv6_small_det` through ONNX Runtime;
3. applies DB post-processing, polygon unclipping, and reading-order sorting;
4. perspective-rectifies each text polygon and rotates tall crops;
5. runs batched `PP-OCRv6_small_rec` inference; and
6. applies CTC decoding and returns text, recognition/detection confidence,
   polygons, bounding boxes, and the source SHA-256.

All inference stays in the local `a3s-use` process. It does not require Python
or PaddlePaddle, does not call an OCR API, and does not transfer image bytes off
the device.

## Commands

```bash
a3s use ocr doctor --json
a3s use ocr extract ./scan.png --json
a3s use mcp serve ocr
```

The standalone development binary accepts the equivalent domain arguments:

```bash
a3s-use-ocr doctor --json
a3s-use-ocr extract ./scan.png --json
a3s-use-ocr serve --mcp
```

Supported inputs are bounded local PNG, JPEG, WebP, GIF, BMP, and TIFF files.
URLs and PDF rasterization are outside this crate.

## Release ownership

This repository owns OCR source, tests, model provenance, Skill content, crate
publication, and platform archives. A3S Use owns the built-in route, capability
projection, component policy, and final product assembly. Releases are joined
only through immutable versions and SHA-256-bound artifacts.

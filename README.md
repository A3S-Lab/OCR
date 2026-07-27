# A3S OCR

`a3s-use-ocr` is the independently maintained, provider-oriented OCR library
used by A3S Use. Its stable boundary is the `OcrProvider` interface, not one OCR
model:

```text
bounded image -> OcrClient -> Arc<dyn OcrProvider> -> provider output
                    |
                    +-> canonical path, media type, size, and SHA-256 evidence
```

`OcrClient` validates and hashes each bounded local image before invoking the
injected provider. A provider owns recognition, readiness, model identity, and
its off-device data policy, but cannot replace the canonical source evidence.
Provider results may include text only or optional confidence, polygon, and
bounding-box evidence.

The default feature set supplies `PpOcrV6Provider`, a local implementation using
ONNX Runtime and the pinned `PP-OCRv6_small` model bundle. It is the default A3S
Use integration, not the architectural limit. Other local or remote providers
can implement `OcrProvider` and be injected as typed objects without changing
the client contract or selecting a backend by a raw string.

## Provider interface

```rust
use a3s_use_ocr::{
    OcrClient, OcrInput, OcrProvider, OcrProviderDescriptor,
    OcrProviderOutput, OcrProviderStatus, Readiness, UseResult,
};
use async_trait::async_trait;

struct MyProvider;

#[async_trait]
impl OcrProvider for MyProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        OcrProviderDescriptor::new("my-provider", "my-engine", false)
            .expect("static provider descriptor")
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        OcrProviderStatus {
            readiness: Readiness::Ready,
            model: Some("my-model".into()),
            model_dir: None,
            message: "ready".into(),
            suggestions: Vec::new(),
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        let _validated_bytes = input.bytes();
        Ok(OcrProviderOutput {
            model: Some("my-model".into()),
            text: "recognized text".into(),
            blocks: Vec::new(),
            warnings: Vec::new(),
        })
    }
}

let client = OcrClient::with_provider(MyProvider)?;
# Ok::<(), a3s_use_ocr::UseError>(())
```

Use `default-features = false` when only the provider-neutral contract and
client are required. Enable `mcp` for a provider-neutral MCP host,
`ppocr-v6` for the local default provider, or `cli` for the bundled application
surface (which assembles both).

## Default PP-OCRv6 provider

The default provider is:

- provider: `pp-ocr-v6`
- engine: `onnx-runtime`
- model bundle: `PP-OCRv6_small`
- source policy: local only

For each image it runs detection, DB post-processing, reading-order sorting,
perspective crop correction, batched recognition, and CTC decoding. It does not
require Python or PaddlePaddle and does not transfer image bytes off device.

The release packages the pinned detection and recognition models. If the model
bundle is absent or damaged, install or repair it explicitly:

```bash
a3s install use/ocr
a3s install use/ocr --force
```

`A3S_OCR_MODEL_DIR` can point development builds at an explicit PP-OCRv6 model
bundle. `A3S_USE_OCR_HOME` overrides its managed model root for packaging,
tests, or an isolated installation. These settings configure the default
provider; custom providers own their own typed configuration.

## Commands

A3S Use assembles the default provider as its reserved built-in `ocr` route:

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

`OcrMcpServer::new` accepts any provider-backed `OcrClient` and projects the
provider's off-device policy into MCP tool annotations. `from_env` selects the
local PP-OCRv6 default for the bundled CLI.

Supported client inputs are bounded local PNG, JPEG, WebP, GIF, BMP, and TIFF
files. URLs and PDF rasterization are outside the current client contract.

## Build

```bash
cargo fmt --all -- --check
cargo test --no-default-features --lib --locked
cargo check --no-default-features --features mcp --locked
cargo test --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo package --locked
```

The library depends on the stable `a3s-use-core` machine contracts. A3S Use
pins an immutable OCR revision when assembling its built-in provider, Skill,
and model assets.

## Release ownership

This repository owns the provider interface, default PP-OCRv6 implementation,
tests, model provenance, Skill content, crate publication, and platform
archives. A3S Use owns the built-in route, chosen default provider, capability
projection, component policy, and final product assembly. Releases are joined
only through immutable revisions and SHA-256-bound artifacts.

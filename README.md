<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S OCR validates a bounded image, routes it through an explicit provider, and returns recognized text with canonical source evidence">
</p>

<p align="center">
  <strong>Provider-oriented OCR for Rust and A3S, with source provenance kept in the result.</strong>
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/OCR/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/OCR/ci.yml?branch=main&amp;style=flat-square&amp;label=CI"></a>
  <a href="https://github.com/A3S-Lab/OCR/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/A3S-Lab/OCR?display_name=tag&amp;sort=semver&amp;style=flat-square&amp;color=2864e8"></a>
  <a href="https://crates.io/crates/a3s-use-ocr"><img alt="a3s-use-ocr on crates.io" src="https://img.shields.io/crates/v/a3s-use-ocr?style=flat-square&amp;color=5420bd"></a>
  <a href="https://docs.rs/a3s-use-ocr"><img alt="docs.rs documentation" src="https://img.shields.io/docsrs/a3s-use-ocr?style=flat-square"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust 1.82 or newer" src="https://img.shields.io/badge/Rust-1.82%2B-a4a8b2?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-contract-ocr-plus-provenance">Contract</a> ·
  <a href="#providers">Providers</a> ·
  <a href="#cli-and-mcp-surfaces">CLI &amp; MCP</a> ·
  <a href="#development">Development</a>
</p>

---

`a3s-use-ocr` is the independently maintained OCR library behind the built-in
A3S Use OCR route. Its stable boundary is [`OcrProvider`](#the-provider-interface),
not a single model.

Every extraction starts with the same client-owned work: resolve a bounded
local image, verify its media type, read it once, and compute canonical source
evidence. Only then are the bytes passed to an injected provider. The provider
may recognize text locally or through an explicitly configured endpoint, but it
cannot replace the source path, media type, size, or SHA-256 recorded by
`OcrClient`.

## Quick start

With A3S Use installed, inspect the configured provider before reading an
image:

~~~bash
a3s use ocr doctor --json
~~~

The diagnostic reports the provider, engine, model, readiness, and
`sendsSourceOffDevice` policy. For the default local provider, install the pinned
model bundle if the diagnostic suggests it, then extract:

~~~bash
a3s install use/ocr
a3s use ocr extract ./scan.png --json
~~~

The standalone binary exposes the same domain operations:

~~~bash
a3s-use-ocr doctor --json
a3s-use-ocr extract ./scan.png --json
a3s-use-ocr serve --mcp
~~~

### Embed the client in Rust

The default feature set includes PP-OCRv6, MCP, and the CLI:

~~~bash
cargo add a3s-use-ocr
~~~

~~~rust
use a3s_use_ocr::{OcrClient, OcrRequest, UseResult};

async fn extract(path: impl Into<std::path::PathBuf>) -> UseResult<String> {
    let client = OcrClient::from_env()?;
    let result = client.extract(OcrRequest { path: path.into() }).await?;
    Ok(result.text)
}
~~~

Use `default-features = false` when an application only needs the neutral
contract and client.

## The contract: OCR plus provenance

The provider owns recognition. `OcrClient` owns the evidence envelope.

| Owned by `OcrClient` | Owned by the provider |
| --- | --- |
| Canonical path, detected media type, byte size, SHA-256 | Recognition text and model identity |
| Input bounds and supported image signatures | Optional confidence, category, polygons, and bounding boxes |
| Provider-output validation | Readiness messages and provider-specific warnings |
| Final `OcrResult` assembly | Declared off-device source policy |

The stable result shape keeps the source next to the OCR evidence:

~~~jsonc
{
  "provider": "unlimited-ocr",
  "engine": "vllm-openai",
  "model": "baidu/Unlimited-OCR",
  "source": {
    "path": "/canonical/path/to/scan.png",
    "mediaType": "image/png",
    "size": 12345,
    "sha256": "..."
  },
  "text": "...",
  "blocks": [
    {
      "page": 1,
      "text": "...",
      "category": {"rawLabel": "title", "role": "title"},
      "boundingBox": {"x": 12, "y": 24, "width": 208, "height": 74},
      "boundingBoxes": [
        {"x": 12, "y": 24, "width": 208, "height": 34},
        {"x": 12, "y": 64, "width": 180, "height": 34}
      ]
    }
  ],
  "warnings": []
}
~~~

Category, confidence, and geometry are optional. `category.rawLabel` preserves
a bounded provider label without declaring the provider taxonomy closed;
`category.role` is a conservative provider-neutral interpretation. Component
boxes retain exact provider geometry, while `boundingBox` is their compatibility
envelope. OCR output is evidence derived from the source, not verified source
text.

## Providers

Provider choice is a typed object, never a raw backend-name switch.

| Provider | Runtime | Source boundary |
| --- | --- | --- |
| `PpOcrV6Provider` | Local ONNX Runtime | Always on device |
| `UnlimitedOcrProvider` | Operator-managed vLLM | Loopback stays local; remote HTTPS is declared off device |
| Custom `OcrProvider` | Defined by the implementation | Required in its descriptor |

### Default: PP-OCRv6

The default A3S integration uses:

- provider ID: `pp-ocr-v6`
- engine: `onnx-runtime`
- pinned bundle: `PP-OCRv6_small`
- transfer policy: local only

Its pipeline is explicit:

~~~text
decode → detect → DB post-process → reading order → perspective crop
       → batched recognition → CTC decode → source-pixel evidence
~~~

The release packages pinned detection and recognition models. Installation and
repair remain explicit:

~~~bash
a3s install use/ocr
a3s install use/ocr --force
~~~

`A3S_OCR_MODEL_DIR` can point development builds at an explicit model bundle.
`A3S_USE_OCR_HOME` overrides the managed model root for packaging, tests, or an
isolated installation. The provider does not require Python or PaddlePaddle.

### Optional: baidu/Unlimited-OCR

Enable the `unlimited-ocr` feature to connect to the official
OpenAI-compatible vLLM serving contract. The Rust crate does not embed the 3B
vision-language model, start Docker or Python, or download its weights.

~~~rust
use a3s_use_ocr::{
    OcrClient, UnlimitedOcrConfig, UnlimitedOcrProvider, UseResult,
};

fn local_unlimited_ocr() -> UseResult<OcrClient> {
    let config = UnlimitedOcrConfig::local("http://127.0.0.1:8000/v1")?;
    OcrClient::with_provider(UnlimitedOcrProvider::new(config)?)
}
~~~

`UnlimitedOcrConfig::local` accepts only loopback endpoints and marks source
bytes as on-device. `UnlimitedOcrConfig::remote` requires HTTPS and marks them
as off-device; `with_bearer_token` adds authentication without exposing the
token through diagnostics or `Debug`.

The provider sends the upstream single-image prompt and no-repeat n-gram
arguments and preserves generated Markdown. It strictly parses both grounding
forms reviewed in the upstream model implementation:

~~~text
<|ref|>title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>text
<|det|>text [x1, y1, x2, y2]<|/det|>text
~~~

Unlimited-OCR coordinates use the closed `0..=999` basis documented by the
[upstream postprocessor](https://huggingface.co/baidu/Unlimited-OCR/blob/07dea832e22aefee32ad281d4b80551282e1c168/modeling_unlimitedocr.py#L62-L111).
A3S OCR resolves the verified input dimensions and maps valid non-image
grounding into typed source-pixel `OcrBlock` evidence. Every valid component
box is preserved in model order and `boundingBox` remains the bounded union for
compatibility. The bounded raw label is retained next to a conservative role:
explicit titles, headings, paragraphs, tables, captions, equations, running
headers/footers, footnotes, page numbers, and code receive matching roles;
other valid labels remain `unknown` rather than being promoted. The upstream
taxonomy is intentionally treated as open.

The adapter evaluates no model text as code, fabricates no confidence, and
emits no geometry for missing, malformed, out-of-range, empty, image-only, or
EXIF-transformed grounding. It never trusts generated image paths. This follows
the upstream loader's EXIF-transpose behavior without mislabeling transformed
coordinates as untransformed source pixels. Degraded grounding remains visible
through one bounded warning while the generated text is preserved.
Diagnostics report `unknown` until extraction checks endpoint reachability
because the synchronous diagnostic interface does not perform network I/O.

<details>
<summary>Official vLLM server recipe</summary>

Use vLLM 0.25.0 or newer and review the upstream
[Unlimited-OCR recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR) before
deployment:

~~~bash
docker run --rm --gpus all --network host --ipc host \
  vllm/vllm-openai:unlimited-ocr \
  baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
~~~

Redirects are disabled so image bytes and credentials are not forwarded to
another origin. Loopback clients also bypass environment-configured HTTP
proxies.

</details>

### The provider interface

`OcrProvider` stays object-safe, `Send + Sync`, and independent of concrete
provider dependencies:

~~~rust
#[async_trait::async_trait]
pub trait OcrProvider: Send + Sync {
    fn descriptor(&self) -> OcrProviderDescriptor;
    fn diagnostic(&self) -> OcrProviderStatus;
    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput>;
}
~~~

Inject an implementation with `OcrClient::with_provider(provider)` or
`OcrClient::from_provider(Arc<dyn OcrProvider>)`. The descriptor must include a
stable provider ID, engine name, and off-device source policy.

## CLI and MCP surfaces

| Surface | Entry point | Provider behavior |
| --- | --- | --- |
| A3S Use | `a3s use ocr ...` | Reserved built-in route; PP-OCRv6 is the current default |
| Standalone CLI | `a3s-use-ocr ...` | Equivalent `doctor`, `extract`, and `serve --mcp` operations |
| Rust library | `OcrClient` | Accepts any typed provider |
| Standard MCP | `OcrMcpServer::new(client)` | Exposes `ocr_doctor` and `ocr_extract` |

`OcrMcpServer` projects the provider's source-transfer policy into the
`ocr_extract` tool annotation. A remote provider therefore remains visible to
the MCP host instead of looking like a local-only read.

## Feature flags

| Feature | Adds |
| --- | --- |
| `ppocr-v6` | Local PP-OCRv6 provider, installer, ONNX Runtime, image pipeline |
| `unlimited-ocr` | Typed HTTP client for an operator-managed Unlimited-OCR vLLM server |
| `mcp` | Provider-neutral standard MCP host |
| `cli` | Standalone CLI; assembles PP-OCRv6 and MCP |
| default | `ppocr-v6`, `mcp`, and `cli` |

## Input and trust boundaries

- Inputs are regular local files between 1 byte and 32 MiB.
- Supported signatures are PNG, JPEG, WebP, GIF, BMP, and TIFF.
- URLs and PDF rasterization are outside the current client contract.
- Providers cannot replace the canonical source evidence created by
  `OcrClient`.
- Pages start at 1; returned confidence values must be finite and between 0 and
  1.
- Provider labels and component-box lists are bounded; a component list must
  exactly agree with its compatibility envelope.
- PP-OCRv6 never transfers source bytes off device.
- Unlimited-OCR remote endpoints are HTTPS-only and explicitly marked as
  off-device.
- Unlimited-OCR source boxes and categories are emitted only from valid,
  bounded `0..=999` grounding and decoded source-image dimensions; malformed
  markers never become boxes or semantic claims.
- Model installation, repair, and external server deployment are never hidden
  inside extraction.

## Development

Run checks from this crate repository, not from the A3S monorepo root:

~~~bash
cargo fmt --all -- --check
cargo test --no-default-features --lib --locked
cargo test --no-default-features --features unlimited-ocr --locked
cargo check --no-default-features --features mcp --locked
cargo test --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo package --locked
~~~

The library depends on the released `a3s-use-core` machine contracts. A3S Use
pins an immutable OCR revision when assembling the built-in route, packaged
Skill, and model assets.

<details>
<summary>Release ownership</summary>

This repository owns the provider interface, default PP-OCRv6 implementation,
Unlimited-OCR adapter, tests, model provenance, Skill content, crate
publication, and platform archives. A3S Use owns the built-in route, chosen
default provider, capability projection, component policy, and final product
assembly. Releases meet through immutable revisions and SHA-256-bound
artifacts.

</details>

## License

Licensed under the [MIT License](LICENSE). See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for model and runtime provenance.

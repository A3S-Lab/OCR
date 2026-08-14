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
  <a href="#responsibility-boundary">Boundary</a> ·
  <a href="#result-contract-ocr-plus-provenance">Contract</a> ·
  <a href="#providers">Providers</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="#cli-and-mcp-surfaces">CLI &amp; MCP</a> ·
  <a href="#development">Development</a>
</p>

---

`a3s-use-ocr` is the independently maintained OCR library behind the built-in
A3S Use OCR route. Its stable boundary is [`OcrProvider`](#the-provider-interface),
not a single model.

Every extraction starts with the same client-owned work: resolve a bounded
local image, verify its media type, read it once, and compute canonical source
evidence. Only then are the bytes passed to an injected provider. A provider
must declare its source-transfer policy and cannot replace the source path,
media type, size, or SHA-256 recorded by `OcrClient`. Both built-in providers
reuse A3S Power's embedded, model-neutral inference substrate; neither enables
Power's HTTP server or opens its own listener.

## Responsibility boundary

A3S OCR recognizes one bounded image and returns OCR evidence. It is not a
document parser.

| A3S OCR owns | Delegated to A3S Power | Outside this repository |
| --- | --- | --- |
| PP-OCRv6 and Unlimited-OCR topology, assets, preprocessing, decoding, labels, confidence, and source-pixel geometry | Typed devices, admission, weight integrity and residency, cancellation, private telemetry, TEE-compatible controls, and execution receipts | Office/PDF page inventory, rendering, cross-page hierarchy, evidence reconciliation, agent planning, and document checkpoints |

PDF rasterization and Office parsing belong to their owning components. A
document-level consumer such as A3S Parser may preserve `OcrResult` blocks and
receipts inside a larger graph, but it must not move OCR model ownership into
the parser. Power remains model-neutral and contains no OCR architecture or
asset.

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

## Staged batch extraction

`OcrClient::extract_batch` accepts stable caller-owned slot IDs and a typed
stage set. It always returns slots in caller order, even when source validation
or one provider stage fails:

~~~rust
use a3s_use_ocr::{
    OcrBatchRequest, OcrBatchSlotId, OcrBatchSlotRequest, OcrClient, OcrStage,
    UseResult,
};

async fn extract_surfaces(client: &OcrClient) -> UseResult<()> {
    let request = OcrBatchRequest::new(
        vec![OcrStage::Preprocessing, OcrStage::Text],
        vec![
            OcrBatchSlotRequest::new(OcrBatchSlotId::new("slide:1")?, "slide-1.png"),
            OcrBatchSlotRequest::new(OcrBatchSlotId::new("slide:2")?, "slide-2.png"),
        ],
    )?;
    let result = client.extract_batch(request).await?;
    assert_eq!(result.slots[0].slot_id.as_str(), "slide:1");
    Ok(())
}
~~~

The provider-neutral stage vocabulary is orientation, preprocessing, layout,
text, table, and formula. A provider descriptor declares the subset it can
complete; unimplemented stages are returned as `unsupported`, never inferred
from text. The compatibility adapter for existing providers supports only the
text stage. PP-OCRv6 currently declares preprocessing and text, where
preprocessing means bounded image decode and canonicalization.

A request contains 1 through 256 unique slots and at most 256 MiB of validated
input bytes in addition to the existing 32 MiB per-image limit. Malformed
request or provider output shapes fail the call. Source, model-load, and stage
execution errors remain attached to their exact slots as completed, partial,
failed, skipped, or unsupported outcomes. The result also carries canonical
provider and per-slot model fingerprints plus digest-only execution receipts;
raw source bytes, tensor values, and local paths are not placed in scheduling
evidence.

## Result contract: OCR plus provenance

The provider owns recognition. `OcrClient` owns the evidence envelope.

| Owned by `OcrClient` | Owned by the provider |
| --- | --- |
| Canonical path, detected media type, byte size, SHA-256 | Recognition text and model identity |
| Input bounds and supported image signatures | Optional confidence, category, polygons, and bounding boxes |
| Provider-output validation | Readiness messages and provider-specific warnings |
| Final `OcrResult` assembly | Declared off-device source policy |

Native results may also contain `executionReceipts`. Each receipt binds a
model family and revision, the exact weight digest, Power runtime/device
identity, and canonical input/output digests. Downstream parsers should
preserve these receipts with the OCR evidence.

The stable result shape keeps the source next to the OCR evidence:

~~~jsonc
{
  "provider": "unlimited-ocr",
  "engine": "a3s-power-native",
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
  "executionReceipts": [
    {
      "schema": "a3s.power.embedded-execution-receipt.v1",
      "model": {"family": "baidu/Unlimited-OCR", "revision": "07dea832...", "weightsSha256": "..."},
      "runtime": {"name": "a3s-power-native", "version": "0.8.0", "device": "metal:0"},
      "input": {"representation": "image-request", "sha256": "...", "byteLength": 12345, "itemCount": 1},
      "output": {"representation": "utf8-text", "sha256": "...", "byteLength": 321, "itemCount": 287}
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

| Provider | OCR-owned implementation | Execution substrate | Source boundary |
| --- | --- | --- | --- |
| `PpOcrV6Provider` | Detection/recognition graphs, image pipeline, DB/CTC postprocessing | Embedded A3S Power | Always on device |
| `UnlimitedOcrProvider` | Vision towers, projector, decoder, tokenizer, generation, and grounding | Embedded A3S Power | Always on device |
| Custom `OcrProvider` | Defined by the implementation | Defined by the implementation | Required in its descriptor |

### Default: PP-OCRv6

The default A3S integration uses:

- provider ID: `pp-ocr-v6`
- engine: `a3s-power-native`
- pinned bundle: `PP-OCRv6_small`
- transfer policy: local only

Its pipeline is explicit:

~~~text
bounded decode → cross-image letterbox → batched detection → per-slot DB
               → identity-bound crop plans → stable width sort
               → cross-image crop batches → CTC decode → ordered evidence
~~~

The OCR-owned release packages pinned detection and recognition SafeTensors
plus their inference configuration. Installation verifies the archive length
and SHA-256, extracts only the four declared files, and records the exact Power
weight digests. Installation and repair remain explicit:

~~~bash
a3s install use/ocr
a3s install use/ocr --force
~~~

Model downloads bound connection setup and stalled reads without imposing a
total transfer deadline, so a healthy slow link can still complete the pinned
archive. Interrupted bodies retry from an exact validated byte range; a server
that ignores the range restarts the staging file instead of appending. The
same bounded retry budget covers transient connection and origin failures. The
complete archive still must match its pinned length and SHA-256 before
activation.

`A3S_OCR_MODEL_DIR` can point development builds at an explicit model bundle.
`A3S_USE_OCR_HOME` overrides the managed model root for packaging, tests, or an
isolated installation. The provider executes reviewed OCR-owned graph plans
through Power's shared admission, device, limit, integrity, cancellation, and
receipt mechanisms. It does not require ONNX Runtime, Python, PaddlePaddle, a
subprocess, an inference service, or a Web listener.

Staged PP-OCRv6 batches reuse an exact, lazily loaded Power model session and
plan deterministic contiguous microbatches from live host/device memory
snapshots. Each admitted microbatch holds one cancellation token, device
permit, and engine lock across its slots and emits a schema-v4 receipt with the
session declaration, plan digest, batch index/count, slot count, and queue
evidence. Detection preprocessing and DB postprocessing use at most 16 bounded
workers and preserve exact slot order. The fast detector bounds the longest
side at 896 pixels, while polygons are mapped back to the immutable source and
recognition crops that original image. An empty fast result on a source with at
least 32 levels of channel variation receives one scalar quality retry with a
4,000-pixel maximum side; both detection receipts remain attached. This retry
protects empty-result quality but is not a guarantee against partial small-text
misses.

Images with different resized dimensions are letterboxed at the top-left of
one normalized-black canvas when every slot retains at least 90% canvas fill.
OCR deterministically splits lower-fill shape outliers and any cohort whose
reviewed peak intermediate would exceed Power's tensor-element limit. Each
compatible cohort contains at most 16 images and executes one dynamic
`[B,3,H,W]` detection graph call. Power validates leading-axis assembly and
output partitions; OCR retains each slot's content extent, excludes padding
from DB postprocessing, and maps polygons through that extent into source
pixels. OCR then flattens detected crops across the admitted images while
retaining exact slot, detection, and reading-order identity. It groups only
identical dynamic recognition canvas widths, chunks each width group into at
most eight crops per graph call, materializes only the active chunk, and
restores blocks and receipts to their source slots. Different widths are never
mixed because PP-OCRv6 recognition has global width context and wider padding
can change decoded text. A failed shared graph call retries its affected crops
through the scalar path so a non-cancellation failure remains isolated;
cancellation still terminates the admitted request.

Recognition no longer materializes the complete 18,710-class probability row
on the host. OCR applies a deterministic model-owned projection on the Power
execution device and transfers `[class index, score, source-finite marker]` for
each CTC time step. The reverse-axis reduction preserves the scalar decoder's
last-class tie rule, while the marker covers every source probability rather
than only the selected score. For the reviewed `[1,40,18710]` output this
reduces host materialization and receipt hashing from 2,993,600 bytes to 480
bytes (**6,236.7x**). The projection revision is part of the model/session
execution identity, and the execution receipt commits to the exact projected
tensor consumed by CTC decoding.

The pinned Power runtime also fuses the reviewed CUDA multiplier-one depthwise
convolutions: 17 detection layers and 14 recognition layers now execute one
F32 kernel per node instead of one device-wide multiply/add sequence per kernel
position. Detection bias is applied in the same kernel after the final term.
Explicit round-to-nearest arithmetic retains the prior accumulation order, and
Power's selected-device parity gate is byte-exact; CPU and unsupported tensor
layouts retain their existing paths. OCR still owns the graph inventory and
end-to-end output parity.

The same pinned Power revision privately fuses adjacent, single-consumer F32
`HardSigmoid`-to-`Mul` channel gates on CUDA. The reviewed OCR plans contain 13
such detection sites and five recognition sites; a graph-inventory test locks
those counts. Each matched `[N, C, 1, 1]` gate over `[N, C, H, W]` replaces the
original four activation passes plus broadcast multiplication with one
byte-exact kernel, removing four launches and four intermediate buffers per
site. Graph topology, receipts, and OCR ownership do not change. CPU and every
unreviewed dtype, shape, broadcast form, or layout retain node-by-node
execution.

The current quality evidence covers the pinned 30-block official image and
clear 8-point and 12-point PDF text rendered at 144 DPI. Five-point synthetic
text did not pass exact publication and is not a supported quality claim. The
named-hardware Parser integration gate is documented by the consuming Parser;
this crate does not turn that workload into a universal OCR throughput claim.

Linux CI installs that exact pinned bundle and executes both reviewed graphs on
the CPU. The gate checks the canonical Power weight digests, exact output
shapes, item counts, and byte lengths for the zero-tensor detection and
recognition fixtures. It then downloads PaddleOCR's SHA-256-pinned
`general_ocr_002` image and executes the complete Rust pipeline: resize,
detection, DB postprocessing, reading-order sort, perspective crops, batched
recognition, CTC decoding, source-coordinate polygons, and eight Power execution
receipts. The 30 output blocks are checked against a reference generated with
Paddle 3.3.1 and PaddleOCR 3.7.0 using explicit text, score, and four-point
coordinate tolerances. The same gate compares one official crop at scalar and
cross-image batch width two, requiring identical text and geometry, recognition
confidence within `0.00001`, one shared recognition receipt, and an exact 2x
input tensor size. Paddle, Python, and ONNX Runtime are not test or runtime
dependencies of this crate.

`a3s-use-ocr-execution-bench` adds a strict, path-free real-provider benchmark
for that pinned image. It separates the first lazy model session from warm
executions, samples process RSS every millisecond, retains both detection and
recognition Power fingerprints, and rejects output drift. The pinned object is
named `.png` upstream but has a JPEG byte signature; source evidence follows
the bytes. Debug or modified-tree reports are diagnostic only. See
[PP-OCRv6 Execution Baseline Protocol](docs/execution-baseline.md) for the
release procedure and claim boundary.

See [Native Inference Architecture](docs/native-inference.md) for the Power/OCR
ownership boundary, model conversion and install integrity, execution receipts,
and TEE/privacy release gates. See [`ROADMAP.md`](ROADMAP.md) for the aligned
Power/OCR/Parser delivery sequence and TurboOCR-derived workstream.

### Optional: baidu/Unlimited-OCR

Enable the `unlimited-ocr` feature to run the reviewed 3B vision-language model
in-process. A3S OCR owns the native Rust model topology, tokenizer,
preprocessing, generation loop, revision pins, and grounding parser. A3S Power
supplies the shared device, admission, weight-integrity, residency, routing,
cancellation, telemetry, and receipt mechanisms.

~~~rust
use a3s_use_ocr::{
    OcrClient, ResidencyBudgetPolicy, UnlimitedOcrConfig,
    UnlimitedOcrProvider,
};

fn local_unlimited_ocr() -> Result<OcrClient, Box<dyn std::error::Error>> {
    let residency = ResidencyBudgetPolicy::new(5_000, 5_000)?
        .with_host_reserve_bytes(2 * 1024 * 1024 * 1024)
        .with_device_reserve_bytes(512 * 1024 * 1024);
    let config = UnlimitedOcrConfig::new("/models/baidu-unlimited-ocr")?
        .with_residency_budget_policy(residency)?
        .with_max_generated_tokens(8_192)?;
    OcrClient::with_provider(UnlimitedOcrProvider::new(config)?)
}
~~~

`UnlimitedOcrConfig::from_env` reads the same local path from
`A3S_UNLIMITED_OCR_MODEL_DIR`. Provider creation is lazy: it performs no model
download, process launch, network request, or socket bind. Session loading
accepts only the pinned upstream revision
`07dea832e22aefee32ad281d4b80551282e1c168`, verifies the exact tokenizer and
processor assets, and asks Power to perform the single full SafeTensors hash
and inventory verification path, including any explicitly configured verified
replicas. The reviewed primary weight file is exactly 6,672,547,120 bytes
with SHA-256
`2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6`.
The non-skippable official-inventory gate resolves only revision
`07dea832e22aefee32ad281d4b80551282e1c168`, verifies Hugging Face's repository
commit plus linked file size and SHA-256, and range-reads the 334,632-byte
SafeTensors JSON header instead of downloading the 6.7 GiB payload. It checks
the pinned small-asset digests, official index, all 2,710 BF16 tensor names,
shapes and byte ranges, the exact 6,672,212,480-byte tensor payload layout, and
an OCR-owned canonical inventory digest. Session loading compares Power's
fully hashed inventory with that same digest before inference. This gate proves
checkpoint identity and topology. A separate local numerical gate executes the
complete official checkpoint and keeps model-output acceptance independent
from inventory acceptance.

The numerical gate downloads the existing SHA-256-pinned PaddleOCR boarding
pass image, derives a fixed 640×528 lossless crop in Rust, and scores all 64
upstream CPU reference tokens through the same KV-cache, no-repeat, and decoder
loop used by production generation. It records every expected-token rank and
logit delta, then performs a second free-running greedy decode. CPU with Apple
Accelerate matches all 64 reference tokens exactly. Metal preserves the first
15 exactly and has at most two rank-2 boundaries with a maximum 0.25 logit
delta; the visible difference is one optional leading punctuation mark and a
three-pixel title-box edge. Both paths must return the same three `header`,
`title`, and `text` blocks, reviewed text, and source-pixel geometry within that
three-pixel bound. Set `A3S_UNLIMITED_OCR_REQUIRE_EXACT_PARITY=1` when auditing
a backend that is expected to provide full teacher-forced token equality.

The native forward path follows the authoritative upstream implementation:

~~~text
EXIF-aware decode
  → 1024px global view + optional bounded 640px tile grid
  → SAM ViT-B detail tower
  → CLIP-L semantic tower over SAM patch features
  → 2048 → 1280 projector + spatial newline/view separator packing
  → 12-layer DeepSeek-style decoder (64 routed experts, exact top-6)
  → deterministic greedy decode + sliding no-repeat 35-gram
  → bounded Markdown and source-pixel grounding
~~~

One logical extraction holds one Power permit and cancellation token across
the complete vision, projector, decoder, and grounding flow. Routed experts
use Power's exact batch union and private-by-default route telemetry rather
than a second OCR-local cache. Cache residency remains zero by default. A typed,
opt-in `ResidencyBudgetPolicy` asks the selected Power runtime to discover
bounded host/CUDA/Metal capacity and derive the cache bytes from explicit
fractions, reserves, caps, and runtime limits; Metal unified memory is counted
once. Manual cache bytes and automatic budgeting are mutually exclusive.
Capacity snapshots are neither persisted nor added to telemetry or execution
receipts. With either explicit cache mode, bounded expert prefetch overlaps
shared-expert computation and uses Power's LFRU/LRU placement. Dropping the
awaiting recognition future cancels that shared token; the blocking native
worker then stops at its bounded preprocessing, vision, and decoder cancellation points.
The provider emits one final receipt binding the source image digest, reviewed
weight collection, Power device, and user-visible UTF-8 text.

CPU is available with `unlimited-ocr`; `unlimited-ocr-accelerate` enables Apple
Accelerate CPU kernels while preserving BF16 model boundaries. Build with
`unlimited-ocr-metal` for an explicit Apple Metal device or
`unlimited-ocr-cuda` for an explicit NVIDIA CUDA device. Typed device selection
fails closed when the requested accelerator is unavailable. Running inside
Power's TEE deployment retains model integrity, resource bounds, private
telemetry, and receipt guarantees; source bytes and detailed routing data are
never exported by this provider.

The provider applies the upstream single-image prompt and no-repeat n-gram
policy in the native generation loop and preserves generated Markdown. It
strictly parses both grounding forms reviewed in the upstream model
implementation:

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

The implementation evaluates no model text as code, fabricates no confidence, and
emits no geometry for missing, malformed, out-of-range, empty, image-only, or
EXIF-transformed grounding. It never trusts generated image paths. This follows
the upstream loader's EXIF-transpose behavior without mislabeling transformed
coordinates as untransformed source pixels. Degraded grounding remains visible
through one bounded warning while the generated text is preserved. Diagnostics
validate the local asset manifest without doing the 6.7 GiB hash twice; the
first session open completes Power's mandatory full checkpoint verification.

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
`ocr_extract` tool annotation. A custom off-device provider therefore remains
visible to the MCP host instead of looking like a local-only read.

## Feature flags

| Feature | Adds |
| --- | --- |
| `power-runtime` | Model-neutral embedded A3S Power runtime; never enables its server feature |
| `ppocr-v6` | Local PP-OCRv6 provider, native graph plans, installer, image pipeline |
| `benchmark` | PP-OCRv6 real-image cold/warm execution-baseline binary |
| `unlimited-ocr` | Native CPU Unlimited-OCR model, tokenizer, image pipeline, generation, and grounding |
| `unlimited-ocr-accelerate` | Unlimited-OCR plus Apple Accelerate CPU kernels with reviewed BF16 operation boundaries |
| `unlimited-ocr-metal` | Unlimited-OCR plus the Power/Candle Apple Metal device path |
| `unlimited-ocr-cuda` | Unlimited-OCR plus the Power/Candle NVIDIA CUDA device path |
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
- Both built-in providers never transfer source bytes off device.
- Unlimited-OCR source boxes and categories are emitted only from valid,
  bounded `0..=999` grounding and decoded source-image dimensions; malformed
  markers never become boxes or semantic claims.
- Model installation, repair, and checkpoint acquisition are never hidden
  inside extraction.
- The built-in embedded inference boundaries contain no ONNX Runtime, external
  OCR service, HTTP client/server, browser automation, Python runtime,
  subprocess inference, or network listener.

## Development

Run checks from this crate repository, not from the A3S monorepo root:

~~~bash
cargo fmt --all -- --check
cargo test --no-default-features --lib --locked
cargo test --no-default-features --features unlimited-ocr --locked
cargo check --no-default-features --features mcp --locked
cargo test --features unlimited-ocr --locked
cargo clippy --all-targets --features unlimited-ocr --locked -- -D warnings
tools/check_official_ppocr_v6.sh /tmp/a3s-ppocr-v6-gate
tools/check_official_unlimited_ocr.sh /tmp/a3s-unlimited-ocr-gate
# With a complete reviewed checkpoint already present:
tools/check_local_unlimited_ocr_checkpoint.sh /models/baidu-unlimited-ocr
tools/check_local_unlimited_ocr_parity.sh /models/baidu-unlimited-ocr
# On macOS:
cargo check --no-default-features --features unlimited-ocr-metal --locked
cargo package --locked
~~~

The library depends on the released `a3s-use-core` machine contracts. A3S Use
pins an immutable OCR revision when assembling the built-in route, packaged
Skill, and model assets.

The staged PP-OCRv6 integration pins the release-ready A3S Power 0.8.0 revision
`51543d20a5da99187b3d05a382504605d2cfb685`. Source builds and CI execute that
exact Git revision. Package verification additionally resolves the declared
`=0.8.0` registry dependency, so the package gate remains closed until the same
Power release is visible on crates.io. No path or `[patch.crates-io]` override
belongs in this repository.

<details>
<summary>Release ownership</summary>

This repository owns the provider interface, default PP-OCRv6 implementation,
native Unlimited-OCR implementation, tests, model provenance, Skill content,
crate publication, and platform archives. A3S Use owns the built-in route, chosen
default provider, capability projection, component policy, and final product
assembly. Releases meet through immutable revisions and SHA-256-bound
artifacts.

</details>

## License

Licensed under the [MIT License](LICENSE). See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for model and runtime provenance.

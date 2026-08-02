# Native Inference Architecture

A3S OCR owns OCR models while A3S Power owns the shared inference substrate.
This boundary keeps Power model-neutral and prevents every model integration
from rebuilding device selection, admission, resource bounds, weight integrity,
residency, cancellation, telemetry, or execution receipts.

## Ownership

| Concern | Owner |
| --- | --- |
| OCR architecture and reviewed graph plans | A3S OCR |
| Model revision, assets, preprocessing, and postprocessing | A3S OCR |
| Tokenization, generation, and grounding semantics | A3S OCR |
| Tensor execution and typed devices | A3S Power |
| Admission, cancellation, limits, and receipts | A3S Power |
| Weight hierarchy, hardware budgets, and routing telemetry | A3S Power |
| TEE encryption, integrity, signatures, privacy, and attestation | A3S Power |

The `power-runtime` feature depends on A3S Power with default features disabled
and enables only `embedded-inference`. It does not activate Power's server,
HTTP client, model registry, or remote backends. Constructing an OCR provider
does not bind a socket or start another process.

## PP-OCRv6

The PP-OCRv6 implementation contains two OCR-owned static graph plans:
detection and recognition. Each plan is bound to the exact source graph digest,
operator set, SafeTensors inventory, model revision, and canonical Power weight
digest. Power validates the complete plan and inventory before execution.

```text
bounded image
  -> OCR preprocessing
  -> Power detection graph
  -> OCR DB postprocessing and perspective crops
  -> Power recognition graph
  -> OCR CTC decoding and source-coordinate blocks
  -> Power execution receipts carried by OcrResult
```

The pinned runtime bundle is published by A3S OCR and contains only:

```text
det/model.safetensors
det/inference.yml
rec/model.safetensors
rec/inference.yml
```

The installer permits only the pinned GitHub release and release-assets hosts,
checks the exact archive byte length and SHA-256, rejects redirects outside the
host allowlist, and extracts only those four regular files. Duplicate, missing,
oversized, linked, nested, or unknown archive entries fail closed. A schema-v2
receipt binds the bundle and both Power weight digests. Schema-v1 installs are
recognized only so an explicit forced repair can migrate them transactionally.

`tools/pack_ppocr_v6.py` is an offline audit/conversion tool. It verifies the
pinned upstream ONNX containers, preserves their numeric tensors in
SafeTensors, and emits deterministic reviewed plans. ONNX is not a runtime
format and neither ONNX Runtime nor Python appears in the inference path.

`tools/check_official_ppocr_v6.sh` is the non-skippable native execution gate
used by pull-request and release CI. It installs the SHA-256-pinned bundle into
a dedicated runner directory, verifies the four required assets, and runs both
reviewed graphs on Linux CPU. Detection is locked to `[1, 1, 64, 64]` and
recognition to `[1, 40, 18710]`; both fixtures also pin the canonical Power
byte length and item count, then require a repeated execution on the same
runner to reproduce the complete tensor and canonical output digest. A
cross-host bitwise digest is deliberately not claimed because CPU kernels may
use hardware-specific floating-point reduction orders. This proves that the
published weights execute deterministically through Power on the release
runner. It does not replace the remaining real-image parity work against the
pinned upstream implementation.

## Unlimited-OCR

The optional Unlimited-OCR provider is an OCR-owned native Rust implementation
of the upstream model at revision
`07dea832e22aefee32ad281d4b80551282e1c168`. A3S OCR pins the exact model,
tokenizer, processor, tensor inventory, weight byte length, and raw weight
SHA-256. Power owns all full SafeTensors hashing, including replica
verification, and exposes the verified collection through one shared
`WeightHierarchy`.

```text
bounded source image
  -> EXIF-aware decode and Pillow-compatible normalized global/tiled views
  -> SAM ViT-B (windowed/global attention and relative positions)
  -> CLIP-L over SAM patch embeddings
  -> OCR-owned projector and spatial token assembly
  -> DeepSeek-style MHA/MoE decoder
  -> deterministic n-gram-constrained generation
  -> bounded Markdown and 0..=999 grounding projection
  -> one Power receipt over source image and visible UTF-8 output
```

The vision tower contains the reviewed 1024-pixel global view and optional
640-pixel tile grid. Its RGB bicubic coefficients, antialiasing, fixed-point
quantization, aspect rounding, and centering match the reviewed Pillow path.
The tower then applies SAM absolute/relative position interpolation, the 24-layer
CLIP branch, the 2048-to-1280 projector, learned row-newline embeddings, and
the view separator. The 12-layer decoder uses exact MHA, RoPE, a dense first
feed-forward layer, and 11 MoE layers with 64 routed experts and exact top-6
weights. OCR owns this topology and generation behavior; no OCR model or asset
is embedded in Power.

One request holds one Power permit and cancellation token across preprocessing,
both vision branches, projection, all decoder layers, and receipt creation.
Dropping the async recognition future cancels that same token; the blocking
native worker observes it at bounded preprocessing, vision, and decoder
boundaries before releasing the request permit.
Power's Colibri-inspired hierarchy supplies exact routed-expert unions,
bounded prefetch, LFRU/LRU placement, transactional hot sets, verified complete
or partial replicas, opt-in native host/CUDA/Metal budget discovery, unified
memory accounting, and private-by-default routing telemetry. The provider does
not create a second hardware probe, cache, integrity path, router, receipt, or
admission controller.

`UnlimitedOcrConfig` accepts only a local model directory plus typed Power
device, limits, residency, and replica settings. Hardware-aware cache planning
is explicitly enabled with `ResidencyBudgetPolicy`; the zero-cache default does
not probe hardware. Power applies the resulting byte budget to the existing
residency policy, counts Metal unified memory once, and keeps the capacity
snapshot out of automatic persistence, telemetry, and receipts. Manual cache
bytes and automatic budgeting cannot be combined. CPU, CUDA, and Metal are
build features; an explicit unavailable device fails closed. Provider
construction is lazy and never downloads a model, invokes Python, starts a
subprocess, contacts an OCR service, or binds a Web port.

## TEE and privacy invariants

- Source pixels and tensor values are not included in placement telemetry.
- Execution receipts contain digests and dimensions, not tensor contents.
- Detailed route heat is opt-in in Power and is never persisted automatically.
- Hardware capacity discovery is opt-in and snapshots are not exported by OCR.
- Weight validation uses Power's canonical hashing rather than a provider-local
  duplicate implementation.
- Native OCR holds one Power admission permit across all component graphs in a
  logical extraction.
- Unlimited-OCR emits one request-level receipt rather than independent vision,
  projector, and decoder receipts.
- Device choice fails closed when an explicitly requested accelerator is not
  available; execution is never silently sent to a remote service.
- Model acquisition is explicit and does not occur during extraction.

## Release gates

Before changing a pinned model or graph plan, verify:

1. exact upstream revision and source digest;
2. complete tensor name, dtype, and shape inventory;
3. deterministic conversion and canonical Power weight digests;
4. fixture and real-image parity against the pinned source implementation;
5. identical model output with placement optimizations enabled and disabled;
6. cancellation, limit, malformed-plan, and wrong-digest failures;
7. an embedded dependency closure without ONNX Runtime, a Web server, browser
   automation, Python inference, or external OCR services.

The official-bundle CPU graph gate is enforced today. Real-image and
Unlimited-OCR upstream parity evidence remain open acceptance work and must not
be reported as complete until reproducible fixtures are checked in.

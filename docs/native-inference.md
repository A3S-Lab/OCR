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
  -> OCR batch letterbox with per-slot content extents
  -> one Power detection graph call for the admitted image batch
  -> OCR per-slot output slicing, DB postprocessing, and crop identities
  -> stable cross-image width sort and bounded Power recognition calls
  -> OCR CTC decoding, source-coordinate blocks, and ordered restoration
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
runner.

The same gate downloads PaddleOCR's official `general_ocr_002` image only after
checking its 128,713-byte length and SHA-256
`4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4`.
Although the upstream URL ends in `.png`, those pinned bytes have a JPEG
signature; `OcrClient` therefore records `image/jpeg` from content detection.
It runs the complete Rust image pipeline and compares 30 ordered blocks against
a one-time Paddle 3.3.1 / PaddleOCR 3.7.0 reference produced with the exact
PP-OCRv6 small models. Whitespace and one reviewed punctuation boundary are
normalized, recognition confidence may differ by at most 0.065, and every
source polygon coordinate may differ by at most four pixels. The gate still
executes no Paddle, Python, ONNX, browser, service, or Web listener.

`a3s-use-ocr-execution-bench` reuses the same fixture through the public client
and provider boundary. Its schema separates first-session model loading from
warm engine reuse, samples process RSS, binds the two model/Power fingerprints,
and hashes canonical output without serializing recognized text or local paths.
See [PP-OCRv6 Execution Baseline Protocol](execution-baseline.md).

### Staged batches and session ownership

The public staged contract names orientation, preprocessing, layout, text,
table, formula, and seal without requiring every provider to implement every
stage. Descriptors declare a canonical supported-stage set. The client validates
up to 256 caller-owned slot IDs, keeps the existing 32 MiB per-image bound, caps
retained validated inputs at 256 MiB, and reconstructs exact caller order after
the provider returns. Source and execution failures stay on their slots;
malformed provider cardinality, identity, order, stage, or receipt evidence
fails the provider contract globally.

Schema `a3s.ocr.staged-batch.v2` carries structured page-local evidence on an
exact source-image pixel canvas. A completed table stage must return bounded
table regions and may return a validated grid with non-overlapping row/column
spans and optional source-pixel cell regions. A completed seal stage must return
bounded seal regions, distinguish confirmed objects from boundary candidates,
and explicitly name only canvas edges the region actually touches when a mark
is clipped. A boundary candidate must name at least one such edge. Polygon
envelopes, confidence values, IDs, counts, text, and containment are validated
at the client boundary. This contract supplies evidence to Parser without
moving cross-page reconciliation or normalized document geometry into OCR.

PP-OCRv6 implements preprocessing and text and returns table and seal as
unsupported. `DocumentFastOcrProvider` is a separate explicit composition that
adds the table stage and, only when its second pinned bundle is configured, the
seal stage. It requires operator-supplied, SHA-256-pinned model assets and does
not broaden the default provider. PP-OCRv6 staged execution is:

```text
validated slots and exact caller IDs
  -> cancellation-aware bounded decode; corrupt images fail only their slots
  -> exact OCR-owned model/execution/resource declaration
  -> Power model-session pool with finite load and device queues
  -> OCR shape cohorts with at least 90% per-slot canvas fill
  -> deterministic contiguous plan from live host/device memory
  -> one admitted Power microbatch permit across each planned slot group
  -> one dynamic leading-axis detection call and exact ordered output slices
  -> scalar high-resolution retry for visually non-uniform empty detections
  -> bounded cross-slot recognition batches and exact identity restoration
  -> per-slot OCR results plus digest-only Power receipt v4 evidence
```

The pool is local to the injected provider and retains at most two exact
sessions with a 1 GiB aggregate resident-weight declaration. A session permits
one active device execution and at most 32 queued executions. Before invoking
the Power planner, OCR partitions caller-contiguous inputs into at-most-16
model-quality cohorts. A proposed common canvas is accepted only when every
slot retains at least 90% content fill and the detection graph's reviewed peak
intermediate remains within Power's tensor-element limit; a lower-fill or
oversized candidate starts a new cohort. Each cohort still receives a
canonical Power plan, current-pressure revalidation, admission permit, and
receipt, so the OCR rule does not replace the shared scheduler. OCR derives the
cohort canvas from the maximum resized width and height, includes that F32
canvas in every slot's conservative host/device peak declaration, and pads each
smaller image with a black pixel transformed by the exact detection mean and
standard deviation. Power counts Metal unified memory only once.

The detector accepts dynamic `B`, executes the stacked `[B,3,H,W]` tensor once,
and returns `[B,1,H,W]`. The fast detector bounds the longest side at 896
pixels. Tensor construction and DB postprocessing use no more than the 16
admitted slots as bounded workers and restore exact order. Power's model-neutral
leading-axis contract validates assembly, exact order, limits, and one positive
output partition per input. OCR masks every partition to its own content width
and height before DB postprocessing, so padding cannot produce a box and
source-coordinate mapping does not use the larger batch canvas. Polygons map
back to the immutable source, and recognition crops that source rather than the
detector raster. If the fast detector returns no boxes and the source spans at
least 32 values in one color channel, OCR retries that slot once with a scalar
4,000-pixel detector input. Both execution receipts are retained. The heuristic
does not prove that a non-empty result found every small text line.

Recognition flattens the resulting crop plans across that admitted image
batch. Each plan retains its source slot and reading-order detection index.
OCR computes the exact post-rotation recognition width without allocating all
crops, stable-sorts identities by that width, and forms canonical dynamic
`[B,3,48,W]` cohorts with `B <= 8`. One canonical cohort may span no more than
16 pixels between its narrowest and widest canvas. The 320-pixel minimum canvas
makes that at most 5% right padding, while any larger difference starts another
cohort. Adjacent canonical cohorts may share one physical call with `B <= 32`
only when their maximum canvas widths are already equal. Consequently every
crop sees the same tensor shape and values it had before coalescing. Perspective
warps and per-slot resize/normalization run independently on the shared Rayon
pool; indexed collection restores canonical order and byte-exact scalar/batch
tests lock the materialized tensors. This replaces only call fragmentation: an
earlier unbounded mixed-width CUDA diagnostic changed a result below the 0.95
token-F1 gate and remains forbidden. Only active crops are materialized,
decoded blocks are restored by retained identity, and a shared Power receipt
is attached once to every participating slot. If a shared call fails without
cancellation, OCR retries its affected crops through bounded scalar calls to
preserve slot isolation. A cancelled permit is never converted into fallback
work. Static width profiles remain a separate optimization.

The scheduler identity is revisioned with both bounds. On SHA-pinned Parser
rasters, the full six-page table gate retains 71 cells and exactly two table
continuations. The full 29-page rider-seal gate reduces 2,583 detected crops to
138 physical recognition calls while retaining the CUDA text fingerprint,
structured-geometry fingerprint, 12 confirmed seals, and two reconciled
right-boundary fragments. On the named RTX 4090, the current five-run,
alternating-order medians are 1.463 seconds (4.101 pages/s) for the table
document and 5.838 seconds (4.968 pages/s) for the rider-seal document, versus
same-run pre-fusion medians of 1.489 and 6.255 seconds. Current single-run CPU
captures are 46.334 seconds (0.129 pages/s) and 334.596 seconds (0.087 pages/s),
respectively. Whitespace-only CTC results are filtered after inference because
Parser cannot publish empty text blocks; detector-score overlap rules out a
safe confidence prefilter. Broader official-image and corpus certification
remains open.

The reviewed plans also retain an explicit inventory of adjacent,
single-consumer `HardSigmoid`-to-`Mul` channel gates: 13 in detection and five
in recognition. The pinned Power revision recognizes only contiguous F32 CUDA
tensors with equal rank-four shapes or an exact `[N, C, 1, 1]` gate over
`[N, C, H, W]`. It evaluates the two affine stages, ordered clamp, and final
multiplication in one byte-exact kernel instead of five launches. One combined
detection-plus-recognition graph pass therefore avoids 72 launches; actual OCR
work may invoke either graph a different number of times. The optimization is
private to Power's executor, does not rewrite the OCR-owned graph declaration,
and preserves ordinary execution for every unmatched device, dtype, shape,
broadcast form, layout, or multi-consumer value.

The recognition plan separately locks 13 adjacent, single-consumer
`Div`-`Erf`-`Add`-`Mul`-`Mul` chains with three scalar initializers. Power's
private CUDA lowering reads those scalars once during model loading and retains
all five original f32 rounding boundaries in one byte-exact kernel. Each
chain avoids four intermediate buffers, and each recognition graph call avoids
52 launches; the 29-page gate's 138 physical calls avoid 7,176 launches. The
graph declaration, session identity, receipts, CTC projection, and CPU path
remain unchanged.

The pinned official 30-block image and clear 8-point and 12-point PDF text at
144 DPI pass exact consumer gates. Five-point synthetic text does not, so the
fast detector is not evidence for arbitrary small text or scans. Broader
single-image, mixed-Office, and scale corpora remain release requirements.

Normalized-black letterboxing changes convolution boundary context compared
with an independently sized scalar tensor. The official mixed-shape gate
therefore mirrors TurboOCR's batch contract: scalar/batch ASCII-token F1 must
remain at least 0.95, exact slot order and cardinality must hold, and all
geometry must remain source-bounded. It does not claim bit-identical detector
maps across different canvas shapes. The gate also proves that compatible
mixed shapes fuse while a quality outlier falls back to a separate Power plan.

OCR does not own document order or cross-page semantics. A parser may use slot
IDs to bind OCR evidence to exact rendered surfaces, but retry/cache authority,
native/visual reconciliation, cross-page continuation, and document graph
construction remain in A3S Parser.

### Document-fast wired tables

The document-fast table path keeps deterministic admission separate from model
authority:

```text
bounded source image
  -> one row-major dark-pixel pass for horizontal and vertical line runs
  -> intersected wired-region candidates, at most eight per page
  -> exact source crop resized and normalized to [N,3,488,488]
  -> one at-most-16-crop Power encoder batch -> [N,256,96]
  -> OCR-owned additive-attention GRU structure/location decoder
  -> validated row/column spans and model cell quadrilaterals
  -> source-pixel assignment of PP-OCRv6 blocks to one cell at most
```

The embedded graph declaration is an offline conversion of the reviewed split
encoder. Conversion is allowed only for its exact source SHA-256 and I/O names.
Seven control-only reshape values are reduced to identities, and three dynamic
nearest-neighbor size calculations are replaced by the reviewed fixed-488
scales. A zero-input probe gate compares the Power output with the source ONNX
output at ten fixed indices with a `5e-5` absolute tolerance. ONNX Runtime and
Python remain offline audit tools and are not runtime dependencies.

The model bundle is accepted only when every canonical file remains under the
configured root and matches its exact length and SHA-256. The Power session
identity additionally binds the encoder graph, canonical encoder weight-store
digest, decoder blob, and structure dictionary. The table stage chunks crops
without materializing an unbounded tensor set, retains cancellation checks,
and attaches encoder receipts to both the page-local result and batch evidence.

On the retained real `merged-row-table` fixture, pages 2 through 4 each produce
one model-backed fragment. The checked grids are respectively 6 by 6
with 29 cells, 8 by 7 with 25 cells, and 3 by 6 with 17 cells; all published
cells have model geometry. These are fixture-specific correctness gates, not a
general table-accuracy score. OCR does not decide that the three fragments are
one logical table. Borderless-table detection remains unsupported.

### Document-fast seal positions

The optional seal path owns a reviewed `PicoDet-L_layout_3cls` model and lowers
its pre-NMS raw head to the static A3S Power graph contract. The development
converter verifies exact Paddle graph, parameter, configuration, and archive
SHA-256 values, cuts before provider scaling and NMS, writes one
`[N,8500,7]` raw output, and produces byte-identical graph and SafeTensors
assets on repeated runs. Paddle, Python, and external inference runtimes are
not production dependencies.

```text
bounded source page
  -> full-page 640x640 view + fixed left/right edge strips
  -> bounded batches of at most eight views through one Power session
  -> seal-class score filter + host NMS + exact source-pixel projection
  -> confirmed page detections and explicitly clipped boundary candidates
  -> optional predecessor-authorized local edge view
  -> page-local evidence only; Parser reconciles adjacent units
```

Adjacent scanning is closed by construction. A request slot may name only the
immediately preceding slot in the same validated batch. A predecessor edge
candidate may trigger one 64-pixel-wide, 320-through-512-pixel-high view on the
same edge of the current page. Two contained narrow model fragments plus one
model envelope may be fused dimension-wise, but the result remains a boundary
candidate. No color threshold, red-pixel rule, implicit slot-order assumption,
or cross-page promotion exists in OCR.

The retained real rider-seal fixture verifies three independently confirmed
interior seals on page 2, a right-edge candidate on page 1, and a narrow
right-edge candidate on page 2 recovered by the explicit predecessor view.
This is fixture evidence, not a general accuracy or throughput claim. On the
current Windows CPU host, the unoptimized generic Power graph took about 12.0
seconds for the six baseline views and 2.84 seconds for the one follow-up view
in an optimized build; CPU graph optimization remains an open release gate.

## Unlimited-OCR

The optional Unlimited-OCR provider is an OCR-owned native Rust implementation
of the upstream model at revision
`07dea832e22aefee32ad281d4b80551282e1c168`. A3S OCR pins the exact model,
tokenizer, processor, tensor inventory, weight byte length, and raw weight
SHA-256. Power owns all full SafeTensors hashing, including replica
verification, and exposes the verified collection through one shared
`WeightHierarchy`.

Pull-request and release CI independently resolve that exact upstream commit,
check the Hugging Face linked weight size and SHA-256, and range-read only the
8-byte SafeTensors prefix plus its 334,632-byte JSON header. The gate verifies
the official small-asset digests and index, all 2,710 BF16 names, shapes,
contiguous byte ranges, the 6,672,212,480-byte tensor payload, and a canonical
OCR-owned inventory digest. Runtime session loading compares Power's fully
hashed `WeightStore` inventory with that same digest. The metadata gate neither
executes upstream Python nor substitutes for numerical model-output parity.

The separate local numerical gate uses the complete reviewed checkpoint and a
SHA-256-pinned real source image. Rust deterministically center-crops the image
to 640×528, verifies the decoded RGB digest, and losslessly re-encodes it before
inference so the fixture does not depend on a platform JPEG encoder. One shared
decoder loop supports both production greedy selection and test-only teacher
forcing. The gate records expected-token rank and max-logit delta for all 64
upstream CPU reference tokens, then repeats the decode with production greedy
selection and parses its visible grounding.

Apple Accelerate CPU execution matches all 64 tokens exactly. Metal retains a
15-token exact prefix and no more than two rank-2 boundaries with a maximum
0.25 logit delta. Its free-running result differs only by one reviewed leading
punctuation boundary and a title-box edge within three source pixels. Both
devices must preserve the three upstream `header`, `title`, and `text` blocks,
their canonical roles, visible content, component boxes, and compatibility
envelopes. This is a bounded numerical and product-output claim, not a claim
that arbitrary BF16 kernels are bitwise identical across devices.

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
bytes and automatic budgeting cannot be combined. CPU, Apple Accelerate, CUDA,
and Metal are build features; an explicit unavailable device fails closed.
Provider construction is lazy and never downloads a model, invokes Python,
starts a subprocess, contacts an OCR service, or binds a Web port.

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
5. per-slot scalar/batch parity for mixed aspect ratios, padding exclusion,
   source-coordinate projection, and exact caller-order restoration;
6. identical model output with placement optimizations enabled and disabled;
7. cancellation, limit, malformed-plan, and wrong-digest failures;
8. an embedded dependency closure without ONNX Runtime, a Web server, browser
   automation, Python inference, or external OCR services.

The official-bundle CPU graph, PP-OCRv6 real-image parity, Unlimited-OCR
checkpoint/inventory, and local Unlimited-OCR numerical/grounding gates are
implemented today. The 6.7 GiB numerical gate remains local rather than pull-
request CI because the official checkpoint is not downloaded into ordinary CI.
Any new device backend must publish its own expected-token rank/delta and
free-running structured-output evidence before it is reported as accepted.

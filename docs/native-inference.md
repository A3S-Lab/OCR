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
digest. Embedded document-fast graph digests cover the LF-normalized Git blobs,
not a platform-specific checkout transformation. Power validates the complete
plan and inventory before execution.

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

The opt-in page-orientation stage owns one pinned Power graph and preserves the
source canvas unless a quarter-turn group is model-equivariant. Its physical
batch is derived from Power's input-byte and tensor-element limits, the exact
F32 `[N,3,224,224]` layout, and the 256-slot request contract. Non-upright first
classifications alone generate the other three rotations; no file name, page
number, text, hash, or empirical image threshold participates. The retained
29-page RTX 4090 alternating gate reduced the former eight-page-cap median from
581.342 to 518.838 ms (49.885 to 55.894 pages/s) while preserving every
canonical result after execution receipts were cleared.

Quarter-turn verification now maps each 224-by-224 sample through the exact
virtual coordinates of the immutable source image. It no longer allocates
three full-resolution rotated rasters per non-upright first classification.
Generated non-square fixtures compare all four virtual orientations with
materialized `image::imageops` rotations and require bit-exact F32 tensors. On
the current 29-page Orientation+Text+Table+Seal RTX 4090 gate, an ABBA sequence
with four cold-process runs per side measured a 4,183.942 ms baseline median
and a 3,949.546 ms candidate median, a 5.6% latency reduction (6.931 versus
7.343 pages/s). Text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8` and
non-receipt semantic SHA-256
`bdfedd8b50cc1bf2b863e4892ff3344fba116b1e758ebc8c33d1665a92dd7092` were
unchanged. The timed input is decoded raster evidence, not PDF rasterization
or Office reconstruction.

Power's CUDA model-session path subsequently removed cudarc activation events
only where one session owns one device identity and stream and tensors cross
the runtime boundary through bounded host values. Ordinary runtime and
accelerator-mesh paths retain event tracking. No OCR graph, arithmetic, tensor
value, content, file identity, or empirical-shape selector changed. Six
interleaved cold-process samples per side on the same 29-page RTX 4090 gate
reduced median latency from 4,030.936 to 3,672.701 ms (7.194 to 7.896 pages/s)
and p90 from 4,161.525 to 3,895.917 ms. Every run retained the exact text and
non-receipt semantic SHA-256 values above. The remaining distance to the
2.900-second decoded-raster target is about 0.773 seconds, or 21.0% of current
latency; complete PDF parsing and Office reconstruction remain outside this
measurement.

Schema `a3s.ocr.staged-batch.v3` carries structured page-local evidence on an
exact source-image pixel canvas. A completed table stage must return bounded
table regions and may return a validated grid with non-overlapping row/column
spans and optional source-pixel cell regions. A completed seal stage must return
bounded seal regions, distinguish confirmed objects from boundary candidates,
and explicitly name only canvas edges the region actually touches when a mark
is clipped. A boundary candidate must name at least one such edge. Polygon
envelopes, confidence values, IDs, counts, text, and containment are validated
at the client boundary. This contract supplies evidence to Parser without
moving cross-page reconciliation or normalized document geometry into OCR.

Version 3 also adds an optional normalized Text selection window and provider
fingerprint v2 records whether the provider supports it. Support is an
exhaustive geometry contract: detection still consumes the complete immutable
source; every detected block with positive-area source-bounding-box overlap is
selected; recognition consumes the entire selected block; and output geometry
stays on the original canvas. The window cannot affect Table or Seal execution.
No selection decision may inspect source names, page numbers, text, hashes, or
provider/model labels.

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
  -> deterministic detection-cohort canvases for per-slot peak declarations
  -> one contiguous plan across caller slots from live host/device memory
  -> one admitted Power microbatch permit across each planned slot group
  -> bounded shape-cohort detection calls and exact ordered output slices
  -> scalar high-resolution retry for visually non-uniform empty detections
  -> one bounded recognition width plan across all successful cohorts
  -> exact slot/block identity restoration and isolated failures
  -> per-slot OCR results plus digest-only Power receipt v4 evidence
```

The pool is local to the injected provider and retains at most two exact
sessions with a 1 GiB aggregate resident-weight declaration. A session permits
one active device execution and at most 32 queued executions. Before invoking
the Power planner, OCR deterministically derives at-most-16 model-quality
detection cohorts so every candidate can declare the exact shared canvas that
bounds its peak memory. A proposed common canvas is accepted only when its area
times the combined cardinality is no larger than the sum of the existing
cohort work and the candidate's independent canvas area, and when the
detection graph's reviewed peak intermediate remains within Power's
tensor-element limit. Otherwise the candidate starts a new cohort. This exact
comparison contains no fill percentage or corpus-derived threshold. Those
cohort boundaries do not create separate admission plans. Power plans caller-contiguous slots, revalidates
current pressure, and issues one permit and receipt per admitted microbatch.
Inside that permit OCR recomputes the same bounded detection partitions; a
planner split can only make their canvases smaller. OCR derives each cohort
canvas from the maximum resized width and height, includes that F32 canvas in
every slot's conservative host/device peak declaration, and pads each smaller
image with a black pixel transformed by the exact detection mean and standard
deviation. Power counts Metal unified memory only once.

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

Recognition flattens the resulting crop plans across every successful
detection cohort in that admitted image microbatch. Detection graph calls stay
shape-bounded, but their boundaries no longer fragment recognition. Each plan
retains its source slot and reading-order detection index. OCR computes the
exact post-rotation recognition width without allocating all
crops, stable-sorts identities by that width, and forms canonical dynamic
`[B,3,48,W]` cohorts with `B <= 8`. Only crops whose tensor widths are exactly
equal can enter the same cohort. Adjacent canonical cohorts may share one
physical call with `B <= 128` only while that exact width remains equal. The
complete input-plus-classifier reservation and Power tensor-element limit
derive a width-specific cap that can only lower `B`; a wide cohort cannot
inherit the default-width batch size.
Consequently every crop sees the same tensor shape and values it had before
coalescing; there is no corpus-tuned padding threshold. Perspective
warps and per-slot resize/normalization run independently on the shared Rayon
pool; indexed collection restores canonical order and byte-exact scalar/batch
tests lock the materialized tensors. This replaces only call fragmentation: an
earlier unbounded mixed-width CUDA diagnostic changed a result below the 0.95
token-F1 gate and remains forbidden. Only active crops are materialized,
decoded blocks are restored by retained identity, and a shared Power receipt
is attached once to every participating slot. If a shared call fails without
cancellation, OCR retries its affected crops through bounded scalar calls to
preserve slot isolation. A cancelled permit is never converted into fallback
work. Before each bounded execution window, independent batches are stably
ordered by descending declared tensor reservation so the largest admitted work
starts first; exact ties preserve canonical order. Static width profiles remain
a separate optimization.

No scheduler branch inspects a source path, file name, page number, recognized
text, document hash, or fixture identity. The only batching inputs are typed
slot/block identities, tensor dimensions, memory declarations, and runtime
limits. A current two-page CPU diagnostic therefore exercises the same path as
any admitted batch: 55 crops across two incompatible detection cohorts retain
28 dynamic widths while physical recognition calls fall from 22 to 19, and
same-machine wall time falls from 7.612 to 7.045 seconds (0.263 to 0.284
pages/s). The strict six-page Text-plus-Table gate falls from 25.430 to 24.428
seconds (0.236 to 0.246 pages/s) while retaining the exact reviewed text and
structure hashes, three fragments, 68 cells, and two unresolved continuation
reviews. The official mixed-shape gate also preserves scalar text, confidence,
geometry, slot identity, failure isolation, and a single admission receipt for
three slots. These are fixture regression measurements, not release-wide
throughput evidence.

Execution identity revision v10 binds exact-width batching, the resource-
bounded 128-crop accelerator cap, the no-additional-canvas-work detection rule,
and largest-declared-work-first recognition order.
On SHA-pinned Parser
rasters, the full six-page table gate retains 71 cells and exactly two table
continuations. The full 29-page rider-seal gate reduces 2,583 detected crops to
138 physical recognition calls while retaining the CUDA text fingerprint,
structured-geometry fingerprint, 12 confirmed seals, and two reconciled
right-boundary fragments. On the named RTX 4090, the current five-run,
alternating-order GELU medians were 1.463 seconds (4.101 pages/s) for the table
document and 5.838 seconds (4.968 pages/s) for the rider-seal document, versus
same-run pre-fusion medians of 1.489 and 6.255 seconds. The later channel-bias
A/B used nine alternating table runs and five seal runs under the current load:
1.387 to 1.340 seconds (4.326 to 4.478 pages/s) and 6.067 to 5.960 seconds
(4.780 to 4.866 pages/s). Every run retained the same text and structured
geometry. The next LayerNorm-tail A/B used the same alternating protocol: nine
table runs fell from a 1.270-second median to 1.215 seconds (4.724 to 4.938
pages/s), while five seal runs were effectively flat at 5.848 versus 5.840
seconds (4.959 versus 4.966 pages/s). Current single-run CPU captures are
46.334 seconds (0.129 pages/s)
and 334.596 seconds (0.087 pages/s), respectively. Whitespace-only CTC results
are filtered after inference because Parser cannot publish empty text blocks;
detector-score overlap rules out a safe confidence prefilter. Broader
official-image and corpus certification remains open.

The 2026-08-23 current-tree A/B isolates the scheduler cap while retaining the
same graph, models, rasters, and four requested stages. Alternating three-run
RTX 4090 medians were 6.889 seconds (4.210 pages/s) for `B <= 32` and 6.472
seconds (4.481 pages/s) for resource-bounded `B <= 128`, a 6.0% latency and
6.4% throughput improvement. Text, order, geometry, detection confidence, and
all non-receipt canonical fields were exact across 2,125 published blocks.
Recognition-confidence differences were bounded to `1.704692841e-5`; they do
not feed scheduling, filtering, or another runtime decision. The 128-crop runs
kept one text fingerprint and one full semantic fingerprint across repetitions.
This is decoded-raster Orientation+Text+Table+Seal evidence, not PDF
rasterization, A3S Office reconstruction, or a 10-pages/s fine-parse claim.

After enabling the resource-derived orientation batch as well, three clean
runs of that earlier revision took 6.412, 6.411, and 6.429 seconds; the median
was 6.412 seconds or 4.523 pages/s with stable text and semantic fingerprints.

The newer evidence path batches the seal-text detector without changing its
four-orientation evidence requirement. It groups only exact preprocessed
tensor shapes and derives the maximum physical batch from the live Power
input-byte and tensor-element limits and the public slot contract. On the
29-page rider fixture, 116 scalar graph calls become 12 batches with at most 11
views. A failed batch retries its views through the scalar path so one shared
failure cannot erase unrelated evidence. Clean same-binary RTX 4090 medians
were 8.346 seconds for the scalar path and 7.498 seconds for the batched path
(3.475 versus 3.868 pages/s). All three runs per side retained identical text
and full non-receipt semantic fingerprints. The 10.2% latency reduction is
current-tree optimization evidence, but the absolute median is below the
earlier historical capture and does not close the 10-pages/s target.

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

Recognition also locks 28 convolution/channel-bias activation prefixes: 10
ReLU, 13 error-function GELU, and five gated HardSigmoid multiply windows. The
inventory gate verifies exact F32 `[1,C,1,1]` bias shapes against convolution
output channels, bounded identity chains, and private consumer counts. Power
retains the ordinary convolution backend and folds only the channel addition
into the reviewed activation kernel, using 32-bit indexing after the existing
`u32` launch bound. This removes 28 further launches per recognition call, or
3,864 across the retained 138-call seal gate, without changing OCR topology,
receipts, CPU behavior, or any unsupported fallback.

The recognition plan also locks five adjacent decomposed LayerNorm affine
tails, each expressed as
`Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)`. The inventory verifies a
single-element F32 epsilon initializer, 120-element F32 scale/bias vectors, and
private intermediate consumers. Power retains the two mean reductions,
centering, and squaring, then evaluates the five pointwise nodes in one
byte-exact CUDA kernel. This avoids 20 launches per recognition graph call, or
2,760 across the retained 138-call seal gate, without changing graph identity,
receipts, CTC projection, CPU behavior, or unsupported fallbacks.

Power also recognizes a contiguous F32 `BatchNormalization` whose output is
private to the exact decomposed error-function GELU formula. Constant
normalization channels, scalar activation initializers, use counts, layout,
dtype, and device fully determine eligibility. The combined output pass keeps
the original arithmetic and rounding sequence. A batch-128 recognition profile
measured the matched slice about 37.5% faster, saving 182--192 microseconds; an
alternating 29-page full-stage comparison reduced mean latency from 3,232.997
to 3,156.856 ms with identical text and semantic fingerprints.

At the terminal classifier boundary, Power may pass a rank-three F32
last-two-axis transpose view directly to a contiguous rank-two classifier
matrix through CUDA strided GEMM. The stride proof is generic and checks no
model name, node name, tensor value, source, or measured shape. Unsupported
layouts retain the existing contiguous materialization. Bitwise tests cover
three unrelated matrix geometries. In the complete recognition probe this
removes all 386 matching transpose launches, which totaled 989.750 microseconds
in the retained baseline. Two precommitted interleaved 29-page cohorts retained
2,518 blocks plus exact text and semantic hashes; across nine samples per side,
the candidate reduced mean latency by 0.51% and median latency by 1.13%. The
small result was measured under variable shared-GPU contention and is not a
stable document-throughput claim. The ignored strict rider gate still rejects
the shared semantic fingerprint against its older reviewed golden, which was
not changed for this performance experiment.

The recognition source additionally locks exactly two private
`Add`-`Identity`-`Sigmoid`-`Mul` windows with a rank-one F32 last-axis bias.
After reviewed Identity normalization, Power may execute each exact
`Add`-`Sigmoid`-`Mul` Swish formula in one contiguous CUDA F32 pass. Admission
uses graph topology, exact use counts, rank, last-axis geometry, dtype, device,
layout, cancellation, and declared bounds only; it checks no model name, node
name, tensor value, source, content, fingerprint, or measured shape. A launch-
blocked 29-page trace observed 142 dynamic matches and removed 284 standalone
pointwise launches. Two precommitted interleaved A/B cohorts improved
independently. Across their combined ten samples per binary, mean latency fell
from 3,113.641 to 2,974.654 ms (4.46%), median latency fell from 3,045.915 to
2,950.303 ms (3.14%), and mean throughput rose from 9.337 to 9.757 pages/s.
Every run retained 2,518 blocks plus exact text and semantic hashes. Individual
samples crossed 10 pages/s, but stable throughput and independent semantic/
Office reconstruction acceptance remain open; the older semantic golden was
not changed.

The recognition source also locks nine adjacent `MatMul -> Add` windows whose
MatMul outputs are private, attributes are empty, weights are rank-two F32,
biases are rank-one F32, and bias lengths equal output columns. The existing
terminal classifier projection retains one window. Power may reuse the GEMM
output for the other six bias-only and two bias-plus-Swish windows, but admits
them only from normalized topology, liveness, arbitrary nonempty prefix
geometry, contiguous F32 layout, same CUDA device, cancellation, and declared
bounds. The composed form preserves the retained Swish arithmetic sequence and
removes one allocation/free pair per internal window; it does not reduce the
preceding baseline's kernel count. Rank-two through rank-four and nonzero-
storage-offset CUDA tests are byte-exact.

The first six-sample-per-binary full-stage cohort improved 2.35% by mean and
3.95% by median. The reverse-order four-sample cohort improved only 0.34% by
mean and regressed 0.66% by median. Across all ten samples per binary, mean
latency improved from 3,024.159 to 2,976.892 ms (1.56%), median latency from
2,999.025 to 2,885.866 ms (3.77%), and mean throughput from 9.589 to 9.742
pages/s; seven interleaved pairs favored the candidate and every run retained
2,518 blocks plus exact text and semantic hashes. A noisy isolated-graph series
improved mean from 11.132 to 10.693 ms and 10% trimmed mean from 10.410 to
10.226 ms, while median regressed from 9.634 to 10.128 ms. This is mixed
same-host evidence for deterministic allocation removal, not stable 10-pages/s
or cross-machine evidence. The standalone non-composing predecessor was
rejected because it disabled two already-retained biased-Swish matches; the
Parser negative-result ledger retains that failure rather than adding an OCR
model, file, content, fingerprint, corpus, or observed-shape workaround.

For a leading-axis recognition batch larger than 32, Power keeps direct CUDA
pointwise and spatial convolution on fixed at-most-32 GEMM reduction groups.
Spatial im2col still executes once, and groups write disjoint ranges of one
final contiguous output. The rule depends only on device, dtype, layout,
nonzero geometry, and executor bounds. It prevents later batch items from
changing earlier F32 results: the full 481-node graph at batch 128 matched four
independent partitions bit-for-bit across 95,795,200 values. By contrast, one
unpartitioned launch changed 313 downstream OCR confidence fields, and a sweep
of 41 explicit cuBLAS algorithms changed 8,104 F32 values in the generic
pointwise parity case.

Numerical admission and OCR batch policy remain separate. Single-pass full-
stage measurements across batch sizes 64 through 256 were non-monotonic, with
throughput ranging from 7.061 to 9.593 pages/s, and a four-lane retry regressed
to 9.190 pages/s at batch 32 and 8.848 pages/s at batch 128. Both the larger
public batch and fourth lane remain disabled; no observed model shape or sample
identity is encoded as a selector.

The CPU execution path remains model-neutral. For contiguous multiplier-one
F32 depthwise convolution, Power partitions fresh NCHW output by batch/channel
and preserves the scalar kernel-row/kernel-column FMA sequence. On x86 and
x86-64, a horizontal-stride-one interior uses one eight-output AVX2/FMA row
accumulator only when the host exposes both instructions and a complete
hardware vector fits. The scalar remainder is unchanged and optional bias is
added only after the complete accumulation chain. Other strides, short rows,
architectures, dtypes, and layouts retain the scalar or ordinary graph path.
This selection uses ISA, topology, and tensor geometry only; OCR model names,
recognized text, sample identity, and measured corpus shapes do not participate.
For pointwise matrix products, the pinned `gemm` dependency can select its
x86-v4 microkernel at runtime when CPUID exposes AVX-512F. Other x86 and
non-x86 hosts retain the library's FMA, SIMD, or scalar dispatch. Custom
one-output rows, four-output tiles, and pretransposed weights were removed after
broad-shape measurements showed mixed or negative results; the runtime does not
encode measured model shapes to select them.

On the Xeon w5-2445 final tree, two 21-page Text-window runs retain 368/368
blocks and take 11.398--12.185 seconds (1.723--1.842 pages/s), versus
15.079--15.297 seconds (1.373--1.393 pages/s) for full-page recognition.
Alternating retained FMA and x86-v4 binaries preserve exact output while
reducing full latency by 9.9--13.6% and window latency by 4.6--14.9%. The exact
six-page Text-plus-Table gate takes 5.485--5.547 seconds with x86-v4 versus
6.040--6.287 seconds with FMA. The 29-page Text-plus-Seal gate takes
36.247--36.891 seconds with x86-v4 versus 40.016--41.492 seconds with FMA and
retains reviewed seal positions, but fails its strict Text golden on the
unreviewed
`fb590fe31928f2ba06106fc2cd4282528b75d7917cba266a304785264945b58e`
fingerprint; an AVX-disabled run has the same fingerprint. These are development
diagnostics, not complete fine-parse or release-throughput claims.

Historical fingerprint provenance does not identify an operator regression.
The reviewed CPU result was captured by an older binary whose recognition
planner admitted up to 16 pixels of right padding. Restoring that allowance on
the current tree produces a third fingerprint rather than the old CPU result
and changes a different set of pages. The current exact-width output therefore
remains unreviewed, while the old result remains bound to its historical
execution identity. Exact-width planning is retained because it preserves every
model input value for a graph with global width context. No page correction,
token replacement, source selector, or empirical padding threshold is admitted;
publication requires independent text truth and A3S Office reconstruction.

A current 29-page Text-only trace measures 30.028 seconds. Bounded parallelism
contracts 242.253 seconds of summed recognition graph work to 22.858 wall
seconds and 10.627 seconds of summed detection work to 5.929 wall seconds; crop
and tensor preparation total about 1.017 seconds. One-variable builds that
disabled direct spatial convolution, replaced terminal projection with explicit
graph operations, serialized outer graph-job windows, or disabled CPU
convolution-bias activation retained the same Text fingerprint. Outer
serialization instead regressed to 74.554 seconds. Ten pages/s therefore
requires less model arithmetic, not another queue heuristic. Any lower-precision
or smaller graph is a distinct digest-pinned OCR model revision and must pass
independent text, geometry, table, seal, cross-page, and Office reconstruction
gates before it can be selected.

An operator- and shape-only two-page trace makes the bound concrete. Warmup and
measurement contain 62 recognition graph calls and 33.087 seconds of summed
operator work: pointwise convolution is 16.030 seconds, depthwise convolution
6.560 seconds, and spatial convolution 4.651 seconds. Convolution is 82.3% of
recognition work. The same trace records less than one second of detection graph
work per two-pass pair. It contains no source pixels, decoded text, path, or
fixture identity. Even a zero-cost implementation of every remaining
recognition operation cannot close the 10x target, so further small-op or queue
tuning is not promoted as the main path.

A temporary shape-only Candle CPU matrix probe also rejects implicit half
precision as the next step. Across four broad geometries, F32 took
0.693--0.957 ms, F16 took 1.421--1.774 ms, and BF16 matrix multiplication was
unsupported by the active backend. The probe used no model values, source
pixels, text, paths, or fixture identity and was removed after measurement.
The OCR contract therefore remains explicit F32. A lower-precision revision
must bind a separately quantized graph and faster backend and pass independent
text, geometry, table, seal, cross-page, and Office reconstruction gates.

A worker-occupancy pointwise scheduling candidate was also rejected despite a
strong isolated result. It reduced one content-free batch geometry from
1.654--2.076 ms to 0.885 ms and passed Power's bitwise pointwise tests, but the
same 29-page Text stage took 31.071 seconds between baseline runs of 30.670 and
30.874 seconds. The implementation was reverted; OCR adds no batch-count or
measured-shape selector to preserve the microbenchmark win.

The pinned official 30-block image and clear 8-point and 12-point PDF text at
144 DPI pass exact consumer gates. Five-point synthetic text does not, so the
fast detector is not evidence for arbitrary small text or scans. Broader
single-image, mixed-Office, and scale corpora remain release requirements.

Normalized-black letterboxing changes convolution boundary context compared
with an independently sized scalar tensor. The official mixed-shape gate
therefore mirrors TurboOCR's batch contract: scalar/batch ASCII-token F1 must
remain at least 0.95, exact slot order and cardinality must hold, and all
geometry must remain source-bounded. It does not claim bit-identical detector
maps across different canvas shapes. The gate also proves both exact-work
branches: shapes share a graph call only when their combined padded tensor adds
no work, while any work-increasing shape uses a separate graph call.

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
  -> locally salient intersected wired-region candidates, at most eight per page
  -> detector-authorized source-edge and exact T-junction proof
  -> unique rectangular partition -> exact source grid, no model execution
  -> unresolved crop resized and normalized to [N,3,488,488]
  -> one at-most-16-unresolved-crop Power encoder batch -> [N,256,96]
  -> OCR-owned additive-attention GRU structure/location fallback decoder
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
digest, decoder blob, and structure dictionary. The table stage chunks only
unresolved crops without materializing an unbounded tensor set, retains
cancellation checks, and attaches encoder receipts to both the page-local result
and batch evidence. Source-backed tables do not fabricate a model receipt.

Source proof uses no document, file, sample, provider, or page-number string.
Missing tracks become negative evidence only within the long-line detector's
observation authority. A widened junction band can promote an edge only when it
reaches both endpoints. A single retained-line terminal can introduce a
primitive orthogonal axis only when an exact centerline crossbar connects the
two neighboring structural intervals; disconnected content does not extend the
endpoint's junction footprint. Repeated terminals close at most the nearest
open component side, and every accepted topology must still be the only
rectangular partition within the fixed search budget.

The retained pages 7 through 24 gate reviews 21 table candidates and now derives
all 21 source grids with zero model fallback. Pages 26, 28, and 29 separately
prove that dense certificate texture produces no wired-table candidate. On the
10-core/20-thread Intel Xeon w5-2445 development CPU, three runs over 29 already
decoded page rasters containing 23 tables took 261.328, 266.309, and 269.421
milliseconds, a median of 108.896 pages/s. This timed region is only candidate
detection and source topology; it excludes decoding, PDF rasterization, text
OCR, seals, Parser work, and Office reconstruction.

On the retained real `merged-row-table` fixture, pages 2 through 4 each produce
one model-backed fragment. The checked grids are respectively 6 by 6
with 29 cells, 8 by 7 with 25 cells, and 3 by 6 with 17 cells; all published
cells have model geometry. These are fixture-specific correctness gates, not a
general table-accuracy score. OCR does not decide that the three fragments are
one logical table. Borderless-table detection remains unsupported.

### Document-fast seal positions

The optional seal path admits only reviewed PicoDet-S 480-pixel or PicoDet-L
640-pixel layout assets by exact weight SHA-256 and byte size, then lowers the
matching pre-NMS raw head to the static A3S Power graph contract. Unknown or
mismatched assets fail closed. The development converter verifies exact Paddle
graph, parameter, configuration, and archive SHA-256 values, cuts before
provider scaling and NMS, writes the profile's bounded raw output, and produces
byte-identical graph and SafeTensors assets on repeated runs. The public
DocumentFast model declaration is composed from this exact admitted profile;
it never labels S evidence as L. Paddle, Python, and external inference runtimes
are not production dependencies.

```text
bounded source page
  -> one full-page exact-profile view (480x480 S or 640x640 L)
     + proof-admitted local/edge views
  -> bounded batches of at most 32 views through one to four Power sessions
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
candidate. Color may admit positive local evidence, but it never suppresses a
full-page view or promotes a cross-page relation; no document-text,
file-name, provider-name, or implicit slot-order rule exists in OCR.

Deduplication treats censored geometry as incomplete evidence rather than a
second object. A complete confirmed region with no clipped edge dominates an
overlapping boundary candidate only when their intersection covers at least the
exact profile's NMS fraction of the smaller region and the complete center lies
inside the boundary extent. Distinct boundary candidates remain separate when
that center test fails. This is a typed status, clipping, geometry, and model-
contract rule; it cannot inspect source identity, page number, text, pixels,
fingerprints, model labels, or corpus membership.

Local-view scheduling removes only work that the decoder is mathematically
unable to publish. A chromatic component is checked against the same immutable
support-region area, minimum-dimension, aspect, and color predicates used after
model projection. An ordinary edge strip must contain at least the decoder's
six required chromatic source pixels; otherwise no subregion can pass. These
checks do not suppress the mandatory full-page view and do not apply to the
achromatic adjacent-page recovery path. On the retained positive fixture they
reduce 67 model views to 44 without changing any confirmed or reconciled box.
The historical 29-page CPU gate completed in 24.255 seconds (1.196 pages/s),
and the 35-page precision gate completed in 21.286 seconds (1.644 pages/s).
The 2026-08-25 CUDA seal-stage diagnostics pass the corrected typed contract in
3.035 seconds (9.555 pages/s) and 2.410 seconds (14.521 pages/s), respectively:
the rider retains eight confirmed plus two reconciled page-1/page-2 boundary
elements, and the precision set retains only its reviewed invitation seal with
no unresolved candidate. Shared WDDM activity was not continuously guarded, so
these timings are diagnostic and complete fine parsing at stable 10 pages/s
remains open.

After orientation, source-canvas and orientation-normalized supplemental views
are dependency-independent. CUDA therefore apportions at most four isolated
sessions by typed remaining-view work and assigns complete existing batches to
the least-loaded worker; CPU and Metal retain one session. A fixed 2 GiB device
reserve plus 4 GiB per session bounds residency, optional replicas fail back to
the admitted set, and batch results are restored to their original order before
receipts or evidence are applied. A strict eight-run four/five-session cohort
kept every normalized cache exact, but five won only 2/4 aggregate pairs and
1/4 treatment-active-document pairs and regressed that document's mean and
median. The fifth session was therefore removed rather than selected from one
favorable unguarded sweep.

Power's per-model-session cuBLAS workspace restores bit-exact concurrent output
without a process-global workspace setting. The workspace guard retains the
same handle and stream, synchronizes on teardown, resets that handle to the
vendor pool, and only then releases the buffer. A strict eight-run follow-up
still found cross-branch refinement pooling slower than the static branch
partition: cache means were 6,210.25/6,307 ms, medians were 6,116/6,316 ms, and
pooling won one of four adjacent pairs. The pooled scheduler and its group state
were removed; all three layout phases remain branch-local. Two final 64-page
runs match the strict control exactly at normalized SHA-256
`8df7df06a2a87950d233ad319a571b2eb07d0698e35a6972c1044fbad8c5ad7b`.

The complete 64-page exact-L diagnostic retains rider `10/10`, invitation
`1/1`, and merged-table `68/68` positioned cells in 6.231 seconds of summed
decoded-raster Orientation+Text+Table+Seal time, about 10.271 pages/s. Parser
retained-cache schema v4 binds the exact extractor identity derived from these
result model declarations and compares it with the live configured backend;
ambiguous v2/v3 captures are not selected by that real-corpus gate. The text
component is derived from the typed PP-OCRv6 detection/recognition profile, and
DocumentFast freezes the exact Power session specification at construction so
post-admission configuration mutation fails closed.
All five Parser/A3S Office reconstructions still report
`fine_parse_ready=false`. The rejected PicoDet-S recall run, pre-fix PicoDet-L
invitation false positives, invalid validations, and all other ineffective
attempts remain in the canonical Parser
[negative-result ledger](https://github.com/contra-sense/agentic-parser/blob/main/docs/ocr-acceleration-plan.md#2026-08-24-onward-performance-negative-result-ledger).

The 2026-08-27 pure-CPU full-small six-page table trace localizes the remaining
wall: 13.101 seconds in recognition, 2.322 seconds in detection, 13.509
milliseconds in table execution, and 0.145 milliseconds in DocumentFast
composition. Recognition executes 147 exact source crops in 74 width cohorts as
147 scalar graphs with two active CPU graphs. This excludes table/projection
micro-optimization as a material route to 10 pages/s. Any smaller or
reduced-precision alternative must be a separately named, digest-pinned OCR
profile with independent accuracy evidence.

Power's following topology-only candidate folds a private contiguous constant
`Reshape` once when the executor is constructed. In both current PicoDet-L call
shapes, the trace moves from 28 executed reshapes to 12 and hands the 16
following channel-bias additions to the existing convolution fusion. Frozen
before/after binaries retain exact current 64-page caches, reconstruction and
visual hashes, table-cell geometry, and seal position/precision evidence. This
is deterministic work elimination, not certified speed: the first A/B used
different tracing, while the first identical trace-free cohort stopped before
sample one when unrelated compilers appeared. Both binaries also retain the
same unresolved older strict six-page text-golden failure, so neither runtime
selection nor a golden update is permitted from this experiment.

Power's subsequent exact private CUDA F32 sigmoid-product lowering preserves an
already materialized broadcast gate and combines the following full-shape
Sigmoid with its single-consumer Mul. The generic contract admits equal shapes,
NCHW per-channel `[N, C, 1, 1]`, and NCHW per-spatial `[N, 1, H, W]`
multipliers using only topology, liveness, dtype, same-device CUDA placement,
contiguity, geometry, cancellation, and element bounds. Equal-shape adjacent
dual Sigmoids may also execute with their Mul in one pass. Every other form
retains ordinary graph execution.

The official PicoDet-L trace proves four active broadcast pairs per call:
Sigmoid accounting changes from `12 executions / 12 source nodes` to `12 / 16`,
while Mul changes from `14 / 18` to `10 / 14`. Generic CUDA offset/broadcast
tests and the complete CPU/CUDA graph suites pass. A fresh 64-page Parser cache
is identical to the retained constant-Reshape baseline after normalizing only
the per-run elapsed field, and all 74 Office reconstruction files are raw-byte
identical, including merged-table and rider-seal positions. The first strict
resource snapshot observed 16% aggregate GPU activity before either frozen
binary started, so the candidate has no accepted performance sample yet.

The combined Power candidate also applies an exact adjacent private
`BatchNormalization -> Sigmoid` edge in the normalization output pass. Swish
matching remains first, the normalized value must have one consumer and no
retained output, and convolution is neither extended nor reordered. Complete
CPU and CUDA graph suites pass. On the same official layout graph,
BatchNormalization accounts for `52 / 152` executions/source nodes, Sigmoid
falls to `8 / 12`, and Mul remains `10 / 14`; this removes four additional
execution boundaries per call. Fresh normalized cache wires for all 64 pages
and all 74 raw Office artifacts remain exact. The emitted 6.001-second OCR sum,
about 10.665 pages/s, was an unguarded correctness diagnostic rather than an
adjacent A/B. A subsequent strict preflight exceeded the GPU-idleness gate, so
the combined candidate still has no accepted performance sample.

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

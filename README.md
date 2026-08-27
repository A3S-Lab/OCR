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
text, table, formula, and seal. A provider descriptor declares the subset it
can complete; unimplemented stages are returned as `unsupported`, never
inferred from text. The compatibility adapter for existing providers supports
only the text stage. PP-OCRv6 currently declares preprocessing and text, where
preprocessing means bounded image decode and canonicalization. It does not yet
claim table or seal detection. The separately constructed
`DocumentFastOcrProvider` composes that text provider with the pinned
SLANet-Plus wired-table model and declares preprocessing, text, and table. When
the separately pinned PicoDet layout bundle is configured, the same explicit
provider also declares seal detection. It remains opt-in because every added
model and its limitations must stay visible to the host. Its public model name
is derived from the exact SHA-256/size-admitted layout profile, so PicoDet-S and
PicoDet-L cannot share an L-labelled provider or retained-evidence identity.
The PP-OCRv6 text component is likewise derived from the typed detection and
recognition profile: full small and small-detection/tiny-recognition have
different composite identities. DocumentFast freezes the exact text session
specification when it is constructed, validates every text-lane output and
batch receipt against that binding, and fails closed if configuration bytes are
changed afterward.

When the separately pinned page-orientation bundle is configured, the explicit
provider classifies the immutable source canvases before Text, Table, and Seal
run in parallel. Power's `max_input_bytes` and `max_tensor_elements`, the exact
`[N,3,224,224]` tensor size, and the 256-slot protocol bound derive the physical
orientation batch; there is no fixed page-count or source-specific branch.
Only pages whose first class is non-upright receive the three quarter-turn
equivariance checks, and an inconsistent group abstains without changing the
source canvas. On the 29-page RTX 4090 gate, alternating medians improved from
581.342 ms at the former eight-page cap (49.885 pages/s) to 518.838 ms at the
Power-derived cap (55.894 pages/s). Canonical outputs were exact after the
expected execution-receipt regrouping.

The three verification turns are sampled directly from immutable source
coordinates instead of materializing full-resolution rotated copies. This is
an execution change only: generated non-square image tests require every F32
preprocessing tensor to be bit-exact with `image::imageops` materialization.
On the current 29-page Orientation+Text+Table+Seal RTX 4090 gate, four
alternating cold-process runs per side reduced median latency from 4,183.942 to
3,949.546 ms (6.931 to 7.343 pages/s). The retained text and non-receipt
semantic SHA-256 values remained exact. This decoded-raster measurement still
excludes PDF rasterization and Office reconstruction and is not a 10-pages/s
fine-parse claim.

The CUDA model sessions now use Power's isolated single-stream lane contract,
which removes only redundant per-activation cross-stream events. General Power
runtimes retain event tracking, and no OCR graph, operator, tensor value,
document identity, content, or measured-shape selector changed. In a fresh
six-sample-per-side interleaved cold-process A/B on the same 29-page RTX 4090
gate, median latency fell from 4,030.936 to 3,672.701 ms and p90 from 4,161.525
to 3,895.917 ms (7.194 to 7.896 pages/s; 8.9% lower median latency and 9.8%
higher throughput). Text and non-receipt semantic SHA-256 values remained
exact. The timed boundary is still decoded raster evidence, so the 10-pages/s
complete fine-parse target remains open.

Staged-batch schema v2 requires every completed table or seal stage to carry a
bounded typed payload on the exact source-image pixel canvas. Table evidence
preserves the detected table region, optional grid dimensions, merged-cell
spans, text, and only the cell geometry actually supplied by the provider. A
cell may additionally retain canonical zero-based indices of text blocks from
the same staged slot when the provider established their ownership in source
pixels. References must be strictly increasing, in range, geometry-backed, and
owned by one cell; the cell text must equal the ordered referenced block bytes.
This is an identity contract, not downstream text matching. Seal
evidence preserves its exact region, optional recognition, canonical canvas
edges when the visible mark is clipped, and whether the model confirmed the
object on that page or retained only a `boundary-candidate`. The client rejects
invalid polygon envelopes, out-of-canvas regions, overlapping or out-of-grid
cells, fabricated clipping, duplicate identities, and unbounded text before
publishing a result.
Cross-page table and rider-seal reconciliation remain Parser responsibilities;
OCR produces page-local evidence and never joins pages itself.

### Opt-in wired-table provider

Set `A3S_OCR_SLANET_PLUS_MODEL_DIR` to a reviewed local bundle with this exact
inventory:

~~~text
encoder/model.safetensors
slanext_wired_decoder.bin
slanext_dict_infer.txt
~~~

Then inject the provider explicitly:

~~~rust
use a3s_use_ocr::{DocumentFastOcrProvider, OcrClient, UseResult};

fn document_fast_client() -> UseResult<OcrClient> {
    OcrClient::with_provider(DocumentFastOcrProvider::from_env()?)
}
~~~

Hosts that need a stable product disposition can call
`DocumentFastOcrProvider::from_env_typed()`. Its
`DocumentFastInitializationErrorKind` distinguishes a missing mandatory model
from invalid configuration without asking the caller to inspect the serialized
`UseError` code. The existing `from_env()` remains a compatibility adapter and
preserves the original `UseError`.

The provider admits conservative wired-table crops from intersecting page
rules, runs the fixed 488-pixel SLANet-Plus encoder through A3S Power only for
unresolved crops, decodes the autoregressive structure and cell quadrilaterals
locally, and assigns PP-OCRv6 text blocks to cells by source-pixel geometry. It
publishes the assigned blocks' exact staged-output indices with each cell so
downstream consumers do not need to repeat an ambiguous bounding-box
assignment.

A line candidate alone is never published as table evidence. Retained tracks
must have local contrast against their parallel background, which rejects dense
security texture without reading page content. Repeated perpendicular terminals
may close only the nearest open component side; internal terminals cannot
invent primitive axes. Before invoking the model, the provider tests every
primitive boundary against immutable source pixels. Missing detector tracks are
negative evidence only over intervals the detector was long enough to observe.
A separator must connect both grid junctions within the detector's spatial
tolerance. A newly discovered T junction additionally requires one exact,
centerline-connected crossbar through both neighboring structural intervals.
Disconnected foreground cannot extend a junction footprint, and damaged
one-sided support remains unknown.

The zero-model path is used only when a bounded exhaustive search finds exactly
one rectangular cell partition consistent with all present and absent edges.
Multiple partitions, damaged rules, or a search beyond the fixed work budget
fall back to SLANet-Plus. This decision uses geometry, detector authority, and
connectivity, never document text, file names, page numbers, sample names, or
provider labels. Candidate detection and source proof execute in the existing
bounded blocking preparation phase, parallel across pages; they do not block
the async scheduler thread.

A non-landscape grid whose wire
counts are strongly transposed is rotated clockwise for inference without
changing its immutable source canvas. Only that inference crop receives a
bounded 4-through-32-pixel source-backed margin. Decoded quadrilaterals are
mapped back to the exact source canvas, and the published table region is the
minimal envelope containing both the detected wire candidate and every
model-backed cell quad.
The local recurrent decoder remains bounded at 1,024 tokens so a
large table can reach its model-produced end token instead of losing its final
rows at the historical 501-token boundary. At model load, the pinned vocabulary
must realize the exact reviewed 50-entry index map and is compiled into typed
row, cell, delimiter, and span tokens. Page decoding and grid construction use
that typed grammar rather than reparsing HTML-like token strings.

The retained rotated-table model gate SHA-pins 18 source pages and requires
their exact candidate counts, orientations, token counts, grid dimensions,
cell counts, and source-canvas geometry. The independent source-proof gate
examines 21 candidates on pages 7 through 24; all 21 now have one reviewed,
unique rectangular partition and none invokes the structure model. Separate
certificate-texture negatives on pages 26, 28, and 29 produce no wired-table
candidates. These are corpus-specific topology and precision gates, not a
general table-accuracy score.

On the 10-core/20-thread Intel Xeon w5-2445 development CPU, the retained
29-page in-memory raster gate contains 23 source-backed tables and zero model
fallbacks. Three default `cargo test --locked` runs of candidate detection plus
source topology took 261.328, 266.309, and 269.421 milliseconds: 110.972,
108.896, and 107.638 pages per second, with a 108.896-pages/s median. The timed
region excludes PNG decoding, PDF rasterization, text OCR, seals, Parser
reconciliation, and Office reconstruction, so it is table-stage evidence only
and not a complete fine-parse throughput claim. The reviewed path currently
covers clockwise quarter-turn scans. Counter-clockwise quarter-turn scans,
borderless tables, and page-local continuation labels remain unsupported by
OCR.

### Opt-in model-backed seal positions

Set `A3S_OCR_PICODET_LAYOUT_MODEL_DIR` to the reviewed local directory that
contains the converted `model.safetensors`. Exact weight SHA-256 and byte size
select the reviewed PicoDet-S 480-pixel or PicoDet-L 640-pixel graph profile;
unknown or mismatched assets fail closed. The checked-in graphs are
deterministic lowerings of the corresponding pinned PaddleOCR raw heads;
production loads neither Paddle nor Python. The exact admitted profile appears
in the provider model declaration and every result. The model owns the `seal`
class and the host performs bounded score filtering and NMS in source-pixel
coordinates. PicoDet-L is the current fine-profile candidate; PicoDet-S remains
a distinct lower-compute capability and failed the reviewed rider recall gate.

The normal page path uses one exact-profile square full-page view for every
valid page: 480 pixels for S or 640 pixels for L.
Chromatic components may add bounded local high-resolution views, while
model-supported pages may add left or right edge refinement; neither path ever
suppresses another page's full-page inference because color is not valid
negative evidence for black, gray, or embossed seals.
Full-page and chromatic views are both source-derived before inference, so they
share one canonical initial batch sequence. Boundary refinements and
predecessor-bound adjacent views remain later phases because their admission
depends on earlier model output. This scheduling changes no view, threshold,
or evidence rule.
Before model admission, a chromatic support region must pass the exact same
area, minimum-dimension, aspect, and source-color predicates that the decoder
will apply after replacing a local model box with that immutable support.
Likewise, an ordinary boundary strip is omitted only when the entire strip has
fewer than the decoder's required six chromatic source pixels. These are
necessary-condition proofs: adjacent-page views retain their achromatic path,
and every valid page still receives full-page inference.
Full-page detections at the reviewed threshold are `confirmed`.
Low-confidence edge evidence is never promoted: it is returned as
`boundary-candidate`, must touch the declared source-canvas edge, and remains
unpublishable as a confirmed object without downstream reconciliation.

The decoder reduces repeated observations of one clipped object before
publishing typed evidence. The observations must name the same exact edge, and
each interval along that edge must contain the other's center. The retained box
is their exact common source intersection and carries the lower confidence.
The perpendicular extent is not used for identity because clipping censored it.
This canonical, input-order-independent reduction does not rank by confidence
or size and does not inspect document text, file names, sample names, or
provider names. It is page-local detector normalization; Parser alone owns
cross-page identity.

A complete confirmed region also dominates one overlapping censored boundary
observation when their intersection covers at least the exact profile's NMS
fraction of the smaller region and the complete center lies inside the censored
extent. The confirmed region must have no clipped edge and the other observation
must declare one. This handles the information asymmetry introduced by clipping
without collapsing distinct boundary objects. The rule uses typed status,
geometry, clipping, and the admitted model contract only; it has no source,
page, text, pixel, fingerprint, model-name, or corpus selector.

For an admitted sequence, a caller may explicitly bind a slot to its immediate
predecessor with `with_adjacent_predecessor`. When the predecessor contains a
bounded edge candidate, OCR runs one additional local view for every distinct
exact source window on the current page. Identical windows share one inference,
and the existing 64-seal page bound also bounds this work. No candidate is
chosen by confidence or box size. This recovered the narrow right-edge fragment
in the retained two-page rider-seal fixture while the second page independently
retained its three interior seals. The adjacency declaration authorizes only
extra page-local evidence collection; Parser still owns cross-page matching,
promotion, and canonical geometry. Seal text recognition and a general seal
accuracy score are not implemented.

A request contains 1 through 256 unique slots and at most 256 MiB of validated
input bytes in addition to the existing 32 MiB per-image limit. Malformed
request or provider output shapes fail the call. Source, model-load, and stage
execution errors remain attached to their exact slots as completed, partial,
failed, skipped, or unsupported outcomes. The result also carries canonical
provider and per-slot model fingerprints plus digest-only execution receipts;
raw source bytes, tensor values, and local paths are not placed in scheduling
evidence.

Staged-batch schema `a3s.ocr.staged-batch.v3` adds an optional normalized Text
selection window and binds support into provider fingerprint v2. A provider
that declares support must run detection on the complete immutable source,
select every detected block whose source bounding box has positive-area
intersection with the window, recognize that whole block, and retain its
original source-canvas geometry. The window therefore reduces recognition
work only; it cannot crop detector input, clip text, change Table or Seal
stages, or route on file names, page numbers, text, hashes, or provider/model
labels. Providers without the capability reject windowed slots.

## Result contract: OCR plus provenance

The provider owns recognition. `OcrClient` owns the evidence envelope.

| Owned by `OcrClient` | Owned by the provider |
| --- | --- |
| Canonical path, detected media type, byte size, SHA-256 | Recognition text and model identity |
| Input bounds and supported image signatures | Optional confidence, category, polygons, bounding boxes, and crop-bound text rotation |
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
envelope. A provider may publish `textRotationMillidegrees` only with one
polygon and no component boxes. It is the canonical clockwise source-image
angle in `[-180000, 180000)` of the recognition x-axis established by the exact
perspective crop, not an angle inferred later from recognized text. OCR output
is evidence derived from the source, not verified source text.

## Providers

Provider choice is a typed object, never a raw backend-name switch.

| Provider | OCR-owned implementation | Execution substrate | Source boundary |
| --- | --- | --- | --- |
| `PpOcrV6Provider` | Detection/recognition graphs, image pipeline, DB/CTC postprocessing | Embedded A3S Power | Always on device |
| `DocumentFastOcrProvider` | PP-OCRv6 text, SLANet-Plus wired-table structure, and optional exact-profile PicoDet-S/L seal positions with typed boundary candidates | Embedded A3S Power | Always on device |
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
weight digests. Embedded SLANet-Plus and PicoDet graph identities hash the
repository's LF-normalized JSON blobs; `.gitattributes` and digest tests reject
platform line-ending drift. Installation and repair remain explicit:

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
snapshots. OCR first derives deterministic detection-cohort canvases only to
declare each slot's conservative peak memory; it does not turn those canvases
into separate admission plans. Each admitted microbatch holds one cancellation
token, device permit, and engine lock across all its slots and emits one
schema-v4 receipt with the session declaration, plan digest, batch index/count,
slot count, and queue evidence. Detection preprocessing and DB postprocessing
use at most 16 bounded workers and preserve exact slot order. The fast detector
bounds the longest side at 896 pixels, while polygons are mapped back to the
immutable source and recognition crops that original image. An empty fast
result on a source with at least 32 levels of channel variation receives one
scalar quality retry with a 4,000-pixel maximum side; both detection receipts
remain attached. This retry protects empty-result quality but is not a
guarantee against partial small-text misses.

Detection candidates share one top-left-aligned normalized-black canvas only
when the combined canvas area multiplied by its batch cardinality is no larger
than the sum of executing the existing cohort and candidate separately. This
exact no-additional-work proof replaces the former 90% canvas-fill threshold.
OCR also splits any cohort whose reviewed peak intermediate would exceed
Power's tensor-element limit. Each compatible cohort contains at most 16
images and executes one dynamic `[B,3,H,W]` detection graph call. Power validates leading-axis assembly and
output partitions; OCR retains each slot's content extent, excludes padding
from DB postprocessing, and maps polygons through that extent into source
pixels. OCR then flattens detected crops across the admitted images while
retaining exact slot, detection, and reading-order identity. Detection cohorts
remain separate graph calls, but they are no longer recognition barriers:
successful crops from every cohort inside the same admitted microbatch enter
one width plan. It stable-sorts dynamic recognition widths into canonical
groups of at most eight crops, but only crops with exactly identical tensor
width may share an inference batch. Adjacent canonical groups are coalesced
only when that exact width remains identical, with an accelerator cap of 128
crops. The planner derives a second, width-specific cap from the complete input
plus classifier reservation and Power's tensor-element limit, so wide inputs
can only reduce the physical batch. No empirical width-difference threshold is used, and batching changes
neither padding nor model input values. Exact-width recognition batches enter
the bounded execution window in descending declared tensor-reservation order;
ties retain canonical order. This reduces the final long-job tail without
inspecting pixels or decoded content. Perspective crops publish the exact recognition x-axis
back into source-image millidegrees, including the quarter-turn applied to tall
crops. Crops and recognition tensors use the shared Rayon worker pool and
restore the same deterministic order; scalar-versus-batch tensor tests are
byte-exact. The planner materializes only the active group and restores blocks
and receipts to their source slots. A failed detection cohort fails only its
own slots. Unbounded width mixing remains forbidden because PP-OCRv6
recognition has global width context and can change decoded text. A failed
shared recognition call retries its affected crops through the scalar path so
a non-cancellation failure remains isolated; cancellation still terminates the
admitted request. Recognition results containing only whitespace are omitted
from public blocks rather than publishing invalid empty evidence; this filter
runs after inference and is not a detector-confidence shortcut. Cohort and
recognition decisions use only tensor dimensions, declared limits, and retained
slot/block identities; they never inspect file names, page numbers, recognized
text, or sample fingerprints.

The historical exact-width release gate uses SHA-pinned Parser rasters. The three-page
cross-page-table fixture retains its exact text fingerprint, `6x6/29`,
`8x7/25`, and `3x6/17` grids/cells, and two continuation edges. The two-page
rider-seal fixture retains its exact text fingerprint, three complete seals,
two IoU-checked right-edge fragments, and one continuation identity. The
full-document gates extend this to all six table pages and all 29 rider-seal
pages: CUDA retains exact text and structured-geometry fingerprints, 71 table
cells, two table continuations, 12 complete seals, and two reconciled boundary
fragments with no unresolved candidate. On the named development RTX 4090,
equal-canvas coalescing plus parallel recognition preprocessing first reduced
the 29-page CUDA median from 8.400 to 6.834 seconds. The current byte-exact GELU
fusion then reduced five-run, alternating-order medians from 1.489 to 1.463
seconds for the six-page table document (4.101 pages/s) and from 6.255 to 5.838
seconds for the 29-page rider-seal document (4.968 pages/s). A subsequent
channel-bias-fusion A/B under the current machine load used nine alternating
runs for the table gate and five for the seal gate: medians fell from 1.387 to
1.340 seconds (4.326 to 4.478 pages/s, 3.4% lower latency) and from 6.067 to
5.960 seconds (4.780 to 4.866 pages/s, 1.8% lower latency), respectively. Every
run retained the same text, table continuation, cell, seal-position, and
boundary-fragment assertions. The subsequent LayerNorm-affine-tail A/B used
the same alternating protocol: the nine-run table median fell from 1.270 to
1.215 seconds (4.724 to 4.938 pages/s, 4.3% lower latency), while the five-run
seal median was effectively flat at 5.848 versus 5.840 seconds (4.959 versus
4.966 pages/s). Current single-run CPU captures took 46.334
seconds (0.129 pages/s) and 334.596 seconds (0.087 pages/s), respectively. The
CUDA result is still below the 10-pages/s complete fine-parse target. These are
fixture-specific correctness and latency diagnostics, not corpus-wide OCR
accuracy or throughput claims.

The current four-stage rider gate additionally compares the former 32-crop
cap with the resource-bounded 128-crop cap in alternating order. On the named
RTX 4090, three 29-page runs per candidate produced medians of 6.889 seconds
(4.210 pages/s) and 6.472 seconds (4.481 pages/s), respectively: 6.0% lower
latency and 6.4% higher throughput. All 2,125 published Text blocks retained
exact text, order, source geometry, detection confidence, and every other
canonical field after execution receipts and recognition confidence were
excluded. Recognition confidence remained finite and in range; 1,727 values
changed because CUDA convolution arithmetic depends on batch shape, with a
maximum absolute difference of `1.704692841e-5`. The 128-crop runs retained
text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`
and a stable full-result semantic SHA-256
`3cea2ae1b7fa15992e212c54f98eb1ea82035fe11f0d2227d8b59a3d9e87dcdc`.
This timed region covers Orientation, Text, Table, and Seal over retained PNG
rasters; it excludes PDF rasterization and A3S Office reconstruction and is
still below the 10-pages/s complete fine-parse target.

With both the 128-crop recognition scheduler and resource-derived orientation
batching enabled, three uncontended runs of that earlier revision took 6.412,
6.411, and 6.429 seconds, a 6.412-second median or 4.523 pages/s. Text and
semantic fingerprints remained stable. This is historical decoded-raster
four-stage evidence, not an end-to-end PDF reconstruction rate.

The current seal-text verifier now groups only identical preprocessed tensor
shapes and derives each physical batch from Power's input-byte and
tensor-element limits plus the public slot bound. The 29-page rider request
therefore executes its 116 required orthogonal views as 12 bounded graph calls
with a maximum batch of 11; a failed shared call falls back to its scalar views
for failure isolation. A clean same-binary RTX 4090 A/B on the newer evidence
path measured scalar runs of 8.164, 8.346, and 9.120 seconds and batched runs of
7.225, 7.791, and 7.498 seconds. Medians fell from 8.346 to 7.498 seconds
(3.475 to 3.868 pages/s), a 10.2% latency and 11.3% throughput improvement.
Every run retained text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`
and full non-receipt semantic SHA-256
`781d5a5e7796f462fa1aeba661e7252ef2edfbde7e1d80bd1de8e6976507b750`.
That median remains below the earlier historical core capture and far
below 10 pages/s, so it supersedes neither release evidence nor the open
complete fine-parse gate.

The subsequent model-neutral runtime optimization removes cudarc activation
events only from isolated, host-bounded model-session CUDA streams. A fresh
six-sample-per-side interleaved cold-process A/B on the complete 29-page
Orientation+Text+Table+Seal raster gate reduced the median from 4.031 to 3.673
seconds (7.194 to 7.896 pages/s) and p90 from 4.162 to 3.896 seconds. All 12
runs retained the exact text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`
and non-receipt semantic SHA-256
`bdfedd8b50cc1bf2b863e4892ff3344fba116b1e758ebc8c33d1665a92dd7092`.
This retained decoded-raster GPU result is not complete PDF rasterization,
Parser reconciliation, or A3S Office reconstruction throughput.

The 2026-08-25 exact-profile checkpoint supersedes it only as a current-tree
diagnostic. PicoDet-S processed all 64 retained pages at about 8.335 pages/s but
returned only four rider seals and was rejected. The first exact PicoDet-L v3
capture restored rider `10/10` but returned three conference-invitation seals
instead of one and was also rejected. After the general complete-versus-censored
geometry correction, all 17 decoder tests, the 29-page rider position gate, and
the 35-page precision gate passed. A fresh L-v3 all-corpus run retained rider
`10/10`, invitation `1/1`, and merged table `68/68` positioned cells. Its five
unguarded decoded-raster Orientation+Text+Table+Seal durations totaled 6.231
seconds, about **10.271 pages/s**. The interactive WDDM GPU was shared and not
continuously process-guarded, so this is not stable 10-pages/s certification;
Parser/A3S Office still reports `fine_parse_ready=false` for every source.

A later Power-only candidate folds private contiguous constant `Reshape` views
once at executor construction. The current PicoDet-L trace executes 12 rather
than 28 reshapes per graph call and exposes exactly 16 existing convolution
channel-bias fusion windows, selected only from topology, constants, layout,
and resource bounds. Frozen baseline/candidate runs retain exact current cache
replay, all reconstruction and visual hashes, rider `10/10`, invitation `1/1`,
merged-table `68/68`, and both seal accuracy gates. Stable speed evidence is
still absent: the first timing comparison used unequal tracing and the first
strict trace-free cohort launched zero samples after its quiet guard observed
unrelated compilers. The older six-page strict table-text golden also fails
identically on both binaries (`d2329b...` actual versus `d675b5...` expected),
so it remains an open accuracy transition rather than being updated for the
candidate.

The next Power-only candidate lowers exact private CUDA F32 sigmoid products.
For the current PicoDet-L broadcast, it retains the already computed
`[N, 1, H, W]` gate and combines only the full-shape Sigmoid with its terminal
Mul; this avoids recomputing the smaller gate's exponential across the three
output channels. The generic Power contract also covers equal-shape products
and `[N, C, 1, 1]` multipliers, with no model, node, source, page, text, pixel,
fingerprint, corpus, or observed-shape selector. The official graph now consumes
four active pairs per call, reducing Mul executions from 14 to 10.

CPU/CUDA graph suites pass, and a fresh source-bound Parser binary reproduces
all five normalized cache files over 64 distinct pages plus all 74 A3S Office
reconstruction artifacts byte-for-byte. The reviewed merged table remains
`68/68` positioned cells and the rider remains `10/10` positioned seals; every
existing `fine_parse_ready=false` gap remains visible. The first post-validation
GPU snapshot was already above the strict idle limit before either binary ran,
so no throughput improvement or stable 10-pages/s claim is attached.

The combined follow-up applies an adjacent private
`BatchNormalization -> Sigmoid` edge in Power's normalization output pass.
Swish retains its existing precedence, and the new path requires one private
consumer without extending or reordering convolution. Complete CPU/CUDA graph
suites pass. On the same official Layout graph, BatchNormalization accounts for
`52 / 152` executions/source nodes, Sigmoid falls to `8 / 12`, and Mul remains
`10 / 14`, removing four more execution boundaries per call. A new 64-page
cache and all 74 Office artifacts remain exact. Their unguarded 6.001-second OCR
sum (about 10.665 pages/s) is a diagnostic only: it was not an adjacent guarded
A/B, and the following strict snapshot exceeded the GPU-idleness gate before
either frozen binary ran.

The latest Power follow-up prepares static BatchNormalization
`[mean, sqrt(variance + epsilon)]` statistics once per graph executor, retaining
the former CPU and CUDA F32 operations and the runtime
`sub -> div -> mul -> add` sequence. Ordinary normalization and its depthwise
and spatial convolution compositions share the prepared tensor. Three direct
GPU tests are byte-exact, complete CPU/CUDA graph suites remain `120/0/9` and
`124/0/39`, and the official Layout operation counts are unchanged as expected.
All five normalized 64-page caches and all 74 Office artifacts match the prior
candidate, including `68/68` merged-table cells and `10/10` rider seals. Its
unguarded 6.570-second OCR sum (about 9.741 pages/s) is not performance evidence;
the later preflight still had active compilers and about 15% shared GPU use.

The retained DocumentFast seal scheduler uses at most four isolated CUDA
layout sessions. Device memory keeps a fixed 2 GiB reserve and budgets 4 GiB
per session; CPU and Metal remain single-session. Source and orientation-
normalized supplemental branches receive sessions from typed view cardinality,
and unchanged batches are assigned by remaining work before results return to
source order. A fifth-session candidate preserved all 64-page caches exactly,
but failed a strict `4,5,5,4,5,4,4,5` cohort: it won only two of four aggregate
pairs and one of four pairs on the only document that admitted the fifth
session, while regressing that document's mean/median from 3,021.0/3,026.0 to
3,074.0/3,099.0 ms. The fifth replica was removed. Every fixed-corpus cache and
test-process repetition exceeded 10 pages/s; complete live PDF parsing and
Office reconstruction remain separate open gates.

Power now gives each CUDA model-session lane its own fixed vendor-sized cuBLAS
workspace. The handle and stream remain alive with that allocation; teardown
synchronizes and resets the same handle to the vendor pool before freeing the
buffer. This restores exact concurrent F32 output without a process-global
workspace switch or a dangling workspace for an independently retained Candle
device. The contract contains no model, graph, source, page, content, geometry,
corpus, or timing selector.

The proposed cross-branch refinement pool was still slower and has been removed.
A strict fixed-order eight-run cohort retained normalized-exact 64-page output,
but static/pooling cache means were 6,210.25/6,307 ms, medians were
6,116/6,316 ms, and pooling won only one of four adjacent pairs. Production
keeps model-contract, refinement, and adjacent-boundary work inside the static
source/supplement partitions, with existing whole-batch least-work scheduling
only inside each branch.

The final Parser integration binary
`58763c3308f0fb8d50abf4decfa1f22d1e86ec4c27b8c9a1d23a60db61717aac`
completed two independent 64-page runs without `CUBLAS_WORKSPACE_CONFIG`.
Both equal the strict static control at normalized cache SHA-256
`8df7df06a2a87950d233ad319a571b2eb07d0698e35a6972c1044fbad8c5ad7b`.
The cache-backed Office tree is path-and-byte identical across all 74 artifacts,
including `68/68` positioned merged-table cells and all `11/11` seals. The
unguarded cache sums and process walls are excluded; the first final strict
timing admission launched no binary because an unrelated compiler chain
appeared, so stable live 10 pages/s is not claimed.

The canonical history of every rejected, inconclusive, invalid, mixed, and
reverted optimization or validation attempt is the Parser
[append-before-removal negative-result ledger](https://github.com/contra-sense/agentic-parser/blob/main/docs/ocr-acceleration-plan.md#2026-08-24-onward-performance-negative-result-ledger).
It includes model-profile failures, invalid compiler/test/shell invocations,
missing A/B evidence, and measured first-principles upper-bound rejections;
successful later evidence never deletes an earlier attempt.

The 2026-08-21 CPU Text-window gate on the Xeon w5-2445 uses 21 structurally
selected pages from the external real-PDF corpus. Against complete-source Text
detection and recognition, the window path retained exactly 368 of 368
positive-intersection blocks with zero missing and zero additional blocks. Two
current-tree warm runs took 15.079--15.297 seconds for full recognition and
11.398--12.185 seconds for windowed recognition (1.373--1.393 versus
1.723--1.842 pages/s, 1.255--1.323x). These runs include exact-width
largest-declared-work-first scheduling, nested-parallel-aware pointwise
execution, removal of a redundant depthwise output clear, Power's generic
eight-lane AVX2/FMA stride-one depthwise interior, and `gemm` runtime dispatch
to its x86-v4 kernel on an AVX-512F host. Alternating retained FMA and x86-v4
binaries reduced full latency by 9.9--13.6% and window latency by 4.6--14.9%
while preserving all 368 blocks exactly. Both dispatches use live ISA and tensor
facts only and retain portable fallbacks. Custom pointwise rows, four-output
tiles, and pretransposed weights were rejected after broad-shape regressions;
no model, document, content, corpus, or empirical-shape threshold was added.
This is a Text-stage routing diagnostic, not complete fine parsing and not a
10-pages/s claim.

Alternating binaries pass the six-page Text-plus-Table gate in 5.485--5.547
seconds with x86-v4 versus 6.040--6.287 seconds with the prior FMA path,
preserving both fingerprints, three table fragments, 68 cells, and two
unresolved continuation reviews. The 29-page Text-plus-Seal diagnostic takes
36.247--36.891 seconds with x86-v4 versus 40.016--41.492 seconds with FMA. That
rejected evidence identity observes eight confirmed seals, four reconciled edge
fragments, and reviewed positions. The strict test still rejects Text fingerprint
`fb590fe31928f2ba06106fc2cd4282528b75d7917cba266a304785264945b58e`
because it is not a reviewed CPU/CUDA golden. An AVX-disabled run produced the
same fingerprint in 51.749 seconds, isolating that accuracy drift from the SIMD
change. The golden remains unchanged and complete fine-parse readiness remains
open.

The fingerprint history is revision evidence, not an optimization target. The
reviewed CPU fingerprint
`91ad46b4501dc82bc7baba87334f735f35ec9eb10ba6b0306213e9cf9dc95ec5`
was emitted by an older binary whose recognition planner could right-pad crops
by up to 16 pixels to share a call. The current exact-width planner repeatedly
emits the still-unreviewed `fb590fe...` result. Temporarily restoring the former
allowance on the current tree emitted a third Text fingerprint,
`191d7c408ec1d091744771f529a62c963bd06a66eab77d573f5ee32a1ed1cc82`,
and changed pages in both directions relative to the old and current results.
This neither validates the current text nor identifies one historical change as
the cause. It does show that forcing the old fingerprint would be corpus
overfitting. Exact-width batching remains because it preserves input values for
a graph with global width context; a new golden requires independent text truth
and A3S Office reconstruction review.

Four one-variable CPU diagnostics retained the current fingerprint: disabling
direct spatial convolution took 41.632 seconds, replacing the terminal
classifier projection with explicit graph operations took 48.659 seconds,
serializing outer graph-job windows took 74.554 seconds, and disabling CPU
convolution-bias activation fusion took 54.675 seconds. None was retained as an
accuracy workaround. A current 29-page Text-only trace takes 30.028 seconds
(0.966 pages/s). Recognition accounts for 242.253 seconds of summed CPU graph
work compressed to 22.858 wall seconds, detection for 10.627 summed seconds
compressed to 5.929 wall seconds, and crop plus tensor preparation for about
1.017 seconds. The next order-of-magnitude path is a separately digest-pinned
lower-compute model or precision revision with independent text, geometry,
table, seal, cross-page, and Office-reconstruction gates. Runtime-selected
silent quantization and document- or corpus-dependent precision are forbidden.

An earlier CPU diagnostic isolates the cross-detection-cohort
recognition boundary. Two real table pages retain 55 crops and 28 exact dynamic
widths while reducing physical recognition calls from 22 to 19; same-machine
wall time fell from 7.612 to 7.045 seconds (0.263 to 0.284 pages/s). The strict
six-page Text-plus-Table gate fell from 25.430 to 24.428 seconds (0.236 to 0.246
pages/s) and retained the exact text SHA-256
`d675b5a37ea9f9fa8666a8a97296d0d651567480dd0a190b88b2bedc19daba55`,
structure SHA-256
`45515d3752806d043920eb3f3d6eaffbc9ecbe7449deca0f4f3094dac9d1cbed`,
three table fragments, 68 cells, and two unresolved continuation reviews. The
official scalar/batch gate also retains text, confidence, geometry, per-source
failure isolation, and one admission receipt for a mixed-shape three-slot
microbatch. These CPU captures are exact-fixture regression evidence, not a
general throughput claim, and remain far below the 10-pages/s complete
fine-parse target.

Historical CPU seal-only integration gates cover 29 positive pages and 35
precision pages. Necessary-condition admission reduced the positive fixture
from 67 to 44 model views; those captures took 24.255 seconds (1.196 pages/s)
and 21.286 seconds (1.644 pages/s). The current typed contract retains eight
reviewed complete rider seals plus only the page-1/page-2 boundary pair, which
Parser reconciles into two positioned elements and one continuation; no
page-26/page-27 relation is fabricated. Current unguarded CUDA seal-stage
diagnostics take 3.035 seconds (9.555 pages/s) for the rider and 2.410 seconds
(14.521 pages/s) for the precision corpus, retaining exactly the invitation's
one reviewed seal and no unresolved precision candidate. These stage-only
numbers do not establish complete fine-parse throughput.

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

Recognition also contains 13 adjacent, single-consumer decomposed GELU chains,
each expressed as `Div`-`Erf`-`Add`-`Mul`-`Mul` with three scalar initializers.
The pinned Power executor captures those scalars once at model load and runs
each chain as one CUDA kernel with explicit division, addition, and
multiplication rounding boundaries. Its byte-exact kernel and full graph gates
remove four launches per chain without rewriting the OCR-owned graph; the 138
recognition calls in the retained 29-page seal gate therefore avoid 7,176
launches. CPU and every unmatched graph, dtype, layout, device, output, or
shared intermediate retain ordinary node-by-node execution.

The recognition graph additionally contains 10 `Conv`-bias-ReLU prefixes, 13
`Conv`-bias prefixes feeding those GELU chains, and five `Conv`-bias prefixes
feeding gated HardSigmoid multiplies. An OCR-owned topology test locks the 28
counts, exact F32 `[1,C,1,1]` bias shapes against convolution output channels,
identity depth, and private-consumer relationships. The pinned Power executor
keeps each convolution on its existing backend and folds the channel addition
into the following byte-exact CUDA activation. Each recognition call avoids 28
more full-tensor launches and buffers; the retained 138-call seal gate avoids
3,864. Launch-bounded 32-bit channel indexing is required for the measured fast
path. CPU and every unreviewed bias, shape, topology, device, dtype, or layout
retain the ordinary graph.

Five decomposed last-axis LayerNorm blocks in recognition retain their two
mean reductions, centering, and squaring, while Power privately fuses each
exact `Add(epsilon)`-`Sqrt`-`Div`-`Mul(scale)`-`Add(bias)` tail. The OCR-owned
inventory test locks the five adjacent private windows, scalar epsilon, and
120-element scale/bias initializers. Explicit F32 rounding boundaries make the
CUDA tail byte-exact with the original five nodes. Each recognition call avoids
20 further launches and intermediate buffers; the retained 138-call seal gate
avoids 2,760. CPU and every unreviewed topology, shape, device, dtype, or layout
retain ordinary execution.

The next Power slice combines a contiguous F32 `BatchNormalization` with its
exact private error-function GELU chain while preserving every graph rounding
boundary. A batch-128 recognition profile reduced the matched normalization
and activation work by about 37.5%, saving 182--192 microseconds. An
alternating 29-page full-stage comparison reduced the baseline mean from
3,232.997 to 3,156.856 ms (2.36%) with unchanged output. Power then lets an
exact rank-three last-two-axis transpose view feed its contiguous rank-two
classifier matrix directly through CUDA strided GEMM. Matching uses only
device, F32 dtype, rank, contiguity, compatible nonzero dimensions, and exact
strides; all other layouts keep materialization. The recognition probe removes
all 386 matching transpose launches (989.750 microseconds in the retained
baseline). Across two precommitted interleaved 29-page cohorts totaling nine
samples per binary, mean latency moved from 3,371.218 to 3,354.116 ms (0.51%)
and median latency from 3,412.336 to 3,373.865 ms (1.13%). Every full-stage run
retained 2,518 blocks, text SHA-256
`8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`, and
non-receipt semantic SHA-256
`49efe252b380179b4385eb114adf1897a9cbe7f1b949a7e9b6037958158cd1ff`.
The latter result is small and contention-sensitive; neither measurement is a
stable 10-pages/s complete fine-parse claim. The ignored strict rider gate
still rejects that shared semantic fingerprint against its older reviewed
golden; the expectation was not relaxed for this optimization.

Power now also combines an exact private last-axis bias addition, Sigmoid, and
self-multiplication into one contiguous CUDA F32 pass. The OCR topology gate
locks exactly two source `Add`-`Identity`-`Sigmoid`-`Mul` windows, while Power
matches only the normalized formula, private use counts, rank, exact last-axis
geometry, dtype, device, layout, cancellation, and declared bounds. A launch-
blocked full-stage trace observed 142 dynamic matches and removed 284
standalone pointwise launches. Two precommitted interleaved 29-page A/B cohorts
improved independently: six samples per binary moved from 3,081.185 to
2,932.748 ms, and a reverse-order four samples per binary moved from 3,162.325
to 3,037.512 ms. Across all ten samples per binary, mean latency fell 4.46%
from 3,113.641 to 2,974.654 ms, median latency fell 3.14%, and mean throughput
rose from 9.337 to 9.757 pages/s. All runs retained 2,518 blocks and the same
text/semantic hashes above. Individual samples crossed 10 pages/s, but the
stable throughput gate and older semantic-golden gate remain open.

The next generic Power path reuses an exact private `MatMul` output for its
last-axis bias and composes the same retained Swish tail when present. OCR's
source-topology gate locks nine adjacent `MatMul -> Add` windows with private
MatMul outputs, empty attributes, rank-two F32 weights, rank-one F32 biases,
and matching output columns. The existing terminal classifier projection owns
one; six internal bias windows and two internal bias-plus-Swish windows are
eligible without exporting model or node identity to Power. Each removes one
allocation/free pair, but no kernel launch relative to the preceding
biased-Swish baseline. Generic rank-two through rank-four and nonzero-offset
CUDA cases are byte-exact.

The first six-sample-per-binary 29-page cohort improved 2.35% by mean and 3.95%
by median. In the reverse-order four-sample cohort, the mean improved only
0.34% and the median regressed 0.66%. Combined means moved from 3,024.159 to
2,976.892 ms (1.56%), combined medians from 2,999.025 to 2,885.866 ms (3.77%),
and mean throughput from 9.589 to 9.742 pages/s; seven of ten pairs favored the
candidate and every run retained the 2,518 blocks and both hashes above. A
noisy isolated-graph series improved mean and 10% trimmed mean but regressed
median from 9.634 to 10.128 ms. The allocation reduction is retained with
mixed timing evidence, not as stable 10-pages/s proof. A preceding standalone
MatMul-bias attempt was rejected before A/B because it intercepted the two Add
nodes needed by the existing Swish fusion and re-exposed two Sigmoid and two
Mul executions per graph. Parser's append-before-removal ledger records this
and every other identified poor, unstable, invalid, or unpromoted attempt; no
model, file, page, text, value, fingerprint, corpus, or measured-shape selector
may recover one.

Recognition calls above 32 items now retain the same CUDA reduction launch
quantum as the reviewed at-most-32 path. Power lowers the full spatial input
once, partitions only pointwise/spatial batched GEMM by a fixed leading-axis
quantum of 32, and writes each partition directly into one final allocation.
This is selected from device, dtype, layout, geometry, and resource bounds;
OCR/model/source/content identity is not visible to the executor. A complete
batch-128 recognition-graph comparison found zero differing bits across all
95,795,200 output values. The earlier unpartitioned batch-128 path changed 313
OCR confidence fields, and every one of 41 explicit cuBLAS algorithms failed a
generic exact-parity case, so neither is admissible.

This numerical fix does not promote a larger OCR batch. One-pass 29-page
throughput for batch sizes `64/96/128/160/192/224/256` was respectively
`9.580/8.950/9.593/7.061/9.339/9.447/9.472 pages/s`; the non-monotonic series
cannot select a production threshold. A four-text-lane follow-up was exact but
slower at both batch 32 (`9.190 pages/s`) and batch 128 (`8.848 pages/s`) and
was reverted. The public recognition default remains 32 and stable 10 pages/s
remains open.

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
recognition, CTC decoding, source-coordinate polygons, one detection receipt,
and at least one recognition receipt. Physical recognition-call count is a
scheduler result rather than an accuracy golden. The 30 output blocks are
checked against a reference generated with
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
policy in the native generation loop and preserves generated Markdown. Apart
from removing the exact terminal control token, it does not rewrite recognized
substrings such as LaTeX-like operators. It
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
only exact reviewed labels for titles, headings, paragraphs, tables, captions, equations, running
headers/footers, footnotes, page numbers, and code receive matching roles;
case variants, punctuation variants, and other valid labels remain `unknown`
rather than being promoted. The upstream
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
| `ppocr-v6-cuda` | PP-OCRv6 and document-fast table/seal inference through Power's NVIDIA CUDA path |
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
`2939668e8ad38e4d3f564144d01c2a5020aa39de`. Source builds and CI execute that
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

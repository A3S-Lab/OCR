# A3S OCR Roadmap

A3S OCR owns bounded image recognition and OCR evidence. It owns PP-OCRv6 and
Unlimited-OCR model topology, assets, preprocessing, postprocessing, decoding,
categories, confidence, and source-pixel geometry. It does not own Office/PDF
rendering, page inventory, cross-page semantics, parser checkpoints, devices,
TEE attestation, or a second inference scheduler.

The performance workstream was reviewed against TurboOCR `main` at
`ed01c3ea2a3c7011bc361c2985215444918409b8` (release `v3.5.0`). TurboOCR is an
algorithm and benchmark reference only. A3S OCR does not depend on its server,
protocol, TensorRT, ONNX Runtime, Python, Paddle runtime, CUDA scheduler, or
model packaging.

## Reference mapping

| TurboOCR mechanism | A3S owner and adaptation |
| --- | --- |
| Mixed-size detection letterbox and one `B <= 8` call | OCR chooses quality-compatible shape cohorts, canvas, valid extents, DB masks, and source-coordinate projection; Power validates generic tensor stacking and slices |
| Flattened recognition crops and width sorting | OCR owns crop identity, width buckets, fill policy, CTC mapping, and restoration to image order |
| Static `(batch, width)` profiles plus dynamic fallback | OCR declares model-owned shape classes; Power validates a digest-bound generic profile and actual fallback evidence |
| Pipeline replicas, finite queues, deadline drop, recycle | Power owns bounded model/device replicas, admission deadlines, health, and receipts; OCR supplies no second pool or watchdog |
| GPU resize/normalize, ROI warp, DB/CTC kernels | OCR owns numerical semantics and reviewed kernels; Power supplies typed devices, limits, and generic execution boundaries |
| HTTP/gRPC service | Not adopted. The library stays embedded and listener-free |

TurboOCR headline throughput and accuracy values are not A3S evidence. A3S
publishes only clean, revision-bound measurements produced by its own public
client and exact model bundle.

## Milestones

### O0 — Provider and evidence foundation

- [x] Object-safe typed providers with explicit off-device transfer policy.
- [x] Bounded source admission and canonical SHA-256 provenance in `OcrClient`.
- [x] Embedded PP-OCRv6 and Unlimited-OCR through model-neutral A3S Power,
      without ONNX Runtime, Python, subprocesses, services, or listeners.
- [x] Bind embedded SLANet-Plus and PicoDet graph identities to LF-normalized
      repository blobs and reject platform-specific line-ending drift.
- [x] Source-pixel polygons/boxes, provider/model fingerprints, and Power
      execution receipts.

### O1 — Staged image batches

- [x] Typed stage requests, stable slot IDs, exact cardinality/order, isolated
      failures, and completed/failed/skipped/unsupported outcomes.
- [x] Exact Power model sessions, finite queues, current-memory microbatch
      plans, one shared permit, cancellation, and receipt-v4 evidence.
- [x] Bounded batches of 1 through 256 source slots, detection microbatches of
      at most 16 images, canonical recognition cohorts of at most eight crops,
      and input-equivalent physical recognition calls of at most 32 crops.

### O2 — Cross-image PP-OCRv6 detection

- [x] Preserve an independent aspect-ratio resize and source extent per image.
- [x] Letterbox mixed shapes onto one top-left-aligned canvas whose padding is
      a black pixel transformed by the exact detection normalization.
- [x] Partition contiguous slots into at-most-16 shape cohorts before Power
      planning; every fused slot must retain at least 90% canvas fill and the
      reviewed peak intermediate must fit Power's tensor-element limit.
- [x] Execute one reviewed dynamic `[B,3,H,W]` detection graph call and split
      `[B,1,H,W]` in exact caller order through Power's generic tensor contract.
- [x] Restrict DB thresholding, contours, scoring, and coordinate projection to
      each slot's valid content extent; padding cannot create a box.
- [x] Include the common F32 canvas in conservative host/device microbatch
      declarations and bind the shape-cohort scheduler into session evidence.
- [x] Bound the fast detector at 896 pixels, retain original-source recognition
      crops, and retry visually non-uniform empty results once at the reviewed
      4,000-pixel quality bound while preserving both receipts.
- [x] Build detection tensors and DB postprocessing with at most 16 bounded
      workers while restoring deterministic slot order and isolating failures.
- [x] Pin Power's one-kernel CUDA lowering for all 17 reviewed detection and 14
      recognition multiplier-one depthwise layers. Detection bias remains one
      final round-to-nearest add in the fused kernel; CPU and unsupported
      layouts retain the existing fallback.
- [x] Lock the 13 detection and five recognition adjacent single-consumer
      `HardSigmoid`-to-`Mul` sites and pin Power's private byte-exact CUDA
      lowering for equal rank-four and exact NCHW channel-gate tensors. Each
      matched site removes four launches; every unreviewed form retains the
      ordinary graph path.
- [x] Cover mixed canvas shapes, padding exclusion, coordinate projection,
      output slicing, bounds, and an official-model scalar/batch gate with the
      TurboOCR-derived ASCII-token F1 floor of 0.95.
- [ ] Persist clean named-hardware scalar/batch reports before enabling a
      release-wide throughput claim.

### O3 — Cross-image recognition width buckets

- [x] Flatten detected crops across admitted images while retaining exact
      `(slot, detection, reading-order)` identity and materializing no more
      than one 32-crop physical recognition batch at a time.
- [x] Stable-sort and batch dynamic canvas widths within a reviewed 16-pixel
      bound (at most 5% padding at the 320-pixel minimum), restoring results
      and shared receipts to original image and block order. SHA-pinned table
      and seal fixtures retain exact text fingerprints while reducing calls
      from 53 to 28 and 33 to 23. Retain an isolated scalar retry for failed
      shared calls and separate every wider cohort.
- [x] Coalesce adjacent canonical eight-crop cohorts only when their final
      canvas width is identical, up to 32 crops, and parallelize independent
      perspective crops and tensor slots without changing any input value or
      output order. The 29-page CUDA rider-seal gate retains its exact text and
      geometry fingerprints while its median falls from 8.400 to 6.834 seconds.
- [x] Omit whitespace-only decoded blocks from public OCR evidence. Confidence
      traces prove blank and nonblank detector scores overlap, so no unsafe
      pre-recognition confidence cutoff is introduced.
- [x] Project reviewed recognition probabilities on the execution device from
      `[N,T,18710]` to exact `[N,T,index/score/finite]` CTC evidence before host
      materialization. Reverse-axis argmax preserves scalar last-class tie
      behavior, and the finite marker still rejects any non-finite source
      probability. The projection revision is bound into session and model
      execution identity.
- [x] Lock the 13 adjacent decomposed GELU chains in the recognition graph and
      pin Power's byte-exact single-kernel CUDA lowering. On the named RTX 4090,
      five-run alternating-order medians fall from 1.489 to 1.463 seconds for
      the six-page table document and from 6.255 to 5.838 seconds for the
      29-page rider-seal document without changing text or structured geometry.
- [x] Lock 10 biased ReLU, 13 biased GELU, and five biased gated-HardSigmoid
      recognition prefixes and pin Power's byte-exact channel-bias CUDA
      lowering. Nine-run table medians fall from 1.387 to 1.340 seconds and
      five-run seal medians from 6.067 to 5.960 seconds while exact table
      continuations, cells, seal positions, and boundary fragments remain
      unchanged.
- [x] Lock five decomposed LayerNorm affine tails and pin Power's byte-exact
      CUDA lowering while retaining the original reductions, centering, and
      squaring. Nine-run table medians fall from 1.270 to 1.215 seconds; the
      five-run seal median remains effectively flat at 5.848 versus 5.840
      seconds with exact text, structure, and geometry fingerprints.
- [ ] Define measured static shape classes and a fill threshold with the exact
      dynamic-width path as fallback; never pad unboundedly merely to hit a
      static class.
- [ ] Prove CTC text/confidence parity, empty-image behavior, partial failures,
      cancellation, and scalar/batch receipt semantics on the official image
      matrix.

### O4 — Device-resident OCR stages

- [ ] Move resize/normalize and perspective ROI warp to reviewed OCR-owned
      device kernels when Power's bounded device-resident handles are ready.
- [ ] Evaluate DB threshold/connected-components kernels behind exact CPU
      parity gates. Keep canonical CPU fallbacks explicit.
- [x] Execute the CTC top-1 projection on CPU/CUDA through Power's model-owned
      graph-output boundary, with scalar parity, official-weight, exact-shape,
      tie, non-finite, and CUDA reviewed-shape gates.
- [ ] Avoid the remaining full detection-map device-to-host copies when only
      bounded maps or boxes are needed.
- [ ] Retain Power admission, TEE/confidential-device policy, cancellation, and
      receipt binding for every fast path.

### O5 — OCR capability depth

- [x] Define staged-batch v2 page-local table and seal evidence with exact
      source canvases, bounded regions, optional cell geometry, merged spans,
      canonical clipped edges, and strict provider-output validation.
- [x] Add an explicit `document-fast-v1` wired-table provider with pinned
      encoder/decoder/dictionary assets, a Power-native batched encoder,
      model-backed cell quadrilaterals, PP-OCRv6 cell text, and exact page-local
      evidence. The retained cross-page fixture checks 6x6/29, 8x7/25, and
      3x6/17 grids on pages 2 through 4.
- [x] Add optional PicoDet-L model-backed seal positions with pinned assets,
      exact source-pixel geometry, confirmed versus boundary-candidate status,
      bounded immediate-predecessor edge views, and retained real rider-seal
      evidence for three interior marks plus two adjacent-page edge fragments.
- [ ] Add explicit orientation, general layout, borderless-table, formula, and
      seal-text providers only with pinned assets, typed outputs, source-pixel
      geometry, and evidence.
- [ ] Preserve provider-native fine geometry; never fabricate line, span, cell,
      or equation boxes from plain text.
- [ ] Keep PP-OCRv6 and Unlimited-OCR behind the same provider/client contract
      without merging their architectures or caches.

### O6 — Release evidence

- [ ] Publish cold/warm, scalar/batch, CPU/Metal/CUDA, and supported
      confidential-GPU captures from clean immutable revisions.
- [ ] Measure throughput, time to first result, p50/p95 latency, peak RSS/device
      memory, queue depth, cancellation, and per-slot failures.
- [ ] Run single-image, mixed-size, dense/sparse text, multi-surface Office, and
      10,000-surface Parser workloads with byte-stable completed resume.
- [ ] Reject a release claim on numerical/output drift, unbounded growth,
      privacy-policy change, stale receipts, or implicit remote execution.
- [ ] Optimize the PicoDet static graph CPU path before making a document-fast
      seal throughput claim; the first retained release build is a correctness
      baseline and does not meet the fine-parse target.
- [ ] Reduce the remaining PP-OCRv6 recognition graph cost. On the retained
      29-page CUDA gate, 138 width-cohort calls dominate the optimized path;
      the latest LayerNorm-tail A/B median is 5.840 seconds (4.966 pages/s),
      not 10 pages/s.

## Cross-repository sequence

1. A3S Power publishes the required model-neutral contract and TEE evidence.
2. A3S OCR pins it and implements model-specific batching and geometry.
3. A3S Parser pins the compatible OCR revision and owns render/OCR pipeline
   windows, persistence, reconciliation, cross-page graphs, and overlays.

An OCR optimization is not complete if it requires Parser to understand tensor
shapes or Power to understand OCR models.

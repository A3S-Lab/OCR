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
- [x] Derive page-orientation batches from Power input-byte/tensor limits and
      the exact `[N,3,224,224]` layout instead of a fixed eight-page cap. The
      alternating 29-page RTX 4090 gate improved from 581.342 to 518.838 ms
      (49.885 to 55.894 pages/s) with exact canonical results after receipt
      regrouping.
- [x] Eliminate full-resolution materialization for orientation-equivariance
      views by sampling exact virtual quarter-turn coordinates. Generated
      non-square fixtures retain bit-exact F32 tensors. Four alternating
      cold-process runs per side on the 29-page full-stage RTX 4090 gate reduced
      median latency from 4,183.942 to 3,949.546 ms (6.931 to 7.343 pages/s)
      with exact text and non-receipt semantic fingerprints.
- [x] Use Power's isolated single-stream CUDA model-session contract to remove
      redundant activation events without changing graphs, arithmetic,
      tensors, documents, content, or measured-shape dispatch. Six interleaved
      cold-process samples per side reduced the 29-page full-stage median from
      4,030.936 to 3,672.701 ms (7.194 to 7.896 pages/s), improved p90 from
      4,161.525 to 3,895.917 ms, and retained exact text and semantic hashes.
- [x] Bounded batches of 1 through 256 source slots, detection microbatches of
      at most 16 images, canonical recognition cohorts of at most eight crops,
      and input-equivalent physical recognition calls capped at 128 crops and
      reduced further by the exact-width tensor reservation.

### O2 — Cross-image PP-OCRv6 detection

- [x] Preserve an independent aspect-ratio resize and source extent per image.
- [x] Letterbox mixed shapes onto one top-left-aligned canvas whose padding is
      a black pixel transformed by the exact detection normalization.
- [x] Derive at-most-16 shape cohorts before Power planning so each slot has an
      exact conservative canvas declaration; execute the same partitions
      inside one admitted multi-slot microbatch. A candidate joins a cohort
      only when combined canvas-area times batch-cardinality does not exceed
      the exact separate work; the reviewed peak intermediate must also fit
      Power's tensor-element limit. No percentage fill threshold remains.
- [x] Execute one reviewed dynamic `[B,3,H,W]` detection graph call and split
      `[B,1,H,W]` in exact caller order through Power's generic tensor contract.
- [x] Restrict DB thresholding, contours, scoring, and coordinate projection to
      each slot's valid content extent; padding cannot create a box.
- [x] Include each detection cohort's common F32 canvas in conservative
      host/device microbatch declarations without turning cohort boundaries
      into separate admission plans.
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

- [x] Flatten detected crops across admitted images and all successful
      detection cohorts inside a microbatch while retaining exact
      `(slot, detection, reading-order)` identity and materializing no more
      than one resource-bounded 128-crop physical recognition batch at a time.
- [x] Stable-sort dynamic canvas widths and batch only exactly equal tensors,
      restoring results and shared receipts to original image and block order.
      No empirical width delta is admitted. Retain an isolated scalar retry
      with the original tensor width after failed shared calls.
- [x] Stable-sort independent exact-width batches by descending declared tensor
      reservation before filling the bounded worker window. Equal work retains
      canonical order; pixels, text, paths, page numbers, hashes, labels, and
      fixture identity never participate. Bind the scheduling contract into
      PP-OCRv6 execution identity revision v9.
- [x] Prove that detection shape partitions are not recognition barriers. A
      two-page CPU gate keeps 55 crops and 28 dynamic widths while reducing
      physical recognition calls from 22 to 19 and wall time from 7.612 to
      7.045 seconds. The strict six-page Text-plus-Table gate falls from 25.430
      to 24.428 seconds with identical text/structure hashes, three fragments,
      68 cells, and two unresolved reviews. The official three-slot
      mixed-shape gate preserves scalar text, confidence, geometry, failure
      isolation, and one admission receipt. Scheduling consults no source path,
      file name, page number, recognized text, or fixture fingerprint.
- [x] Coalesce adjacent canonical eight-crop cohorts only when their final
      canvas width is identical, up to 128 crops subject to Power's tensor
      limit, and parallelize independent
      perspective crops and tensor slots without changing any input value or
      output order. The 29-page CUDA rider-seal gate retains its exact text and
      geometry fingerprints while its median falls from 8.400 to 6.834 seconds.
- [x] Bind the resource-bounded 128-crop scheduler as execution identity v10.
      An alternating 29-page RTX 4090 A/B reduced the four-stage median from
      6.889 to 6.472 seconds (4.210 to 4.481 pages/s). Text, order, geometry,
      detection confidence, and all other non-receipt canonical fields remained
      exact across 2,125 published blocks; recognition-confidence drift was at
      most `1.704692841e-5` and does not drive a runtime branch.
- [x] Batch the four-direction seal-text verifier by exact preprocessed tensor
      shape under Power's input-byte/tensor-element limits and the public slot
      bound, with scalar failure isolation. The 29-page gate reduces 116 graph
      calls to 12 batches of at most 11. Clean same-binary RTX 4090 medians fell
      from 8.346 to 7.498 seconds (3.475 to 3.868 pages/s), while text SHA-256
      `8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8`
      and full non-receipt semantic SHA-256
      `781d5a5e7796f462fa1aeba661e7252ef2edfbde7e1d80bd1de8e6976507b750`
      remained exact. No path, page, text, pixel value, model label, or corpus
      threshold participates in batching.
- [x] Omit whitespace-only decoded blocks from public OCR evidence. Confidence
      traces prove blank and nonblank detector scores overlap, so no unsafe
      pre-recognition confidence cutoff is introduced.
- [x] Extend staged-batch schema v3 with exhaustive normalized Text selection
      and bind support into provider fingerprint v2. Keep detection on the full
      immutable source, recognize every whole detected block with positive-area
      source-box intersection, and preserve original coordinates. The retained
      21-page Xeon w5-2445 CPU gate returns 368/368 blocks with zero missing or
      additional evidence. Two 2026-08-21 final-tree warm runs take
      15.079--15.297 seconds full and 11.398--12.185 seconds windowed
      (1.255--1.323x). Power's AVX2/FMA depthwise row gate and `gemm` x86-v4
      runtime dispatch use only live ISA, topology, and tensor geometry and
      retain scalar or portable fallbacks. Alternating retained binaries show
      9.9--13.6% lower full latency and 4.6--14.9% lower window latency while
      preserving all 368 blocks. Rejected pointwise kernels and pretransposed
      weights were reverted instead of adding corpus or empirical-shape gates.
      This is a Text-stage diagnostic, not complete fine parsing.
- [ ] Independently review the full rider-seal Text output before accepting its
      current CPU latency. The 29-page diagnostic retains eight confirmed
      seals, four reconciled fragments, and reviewed positions in
      36.247--36.891 seconds with x86-v4 versus 40.016--41.492 seconds with
      FMA, but strict Text fingerprint validation rejects
      `fb590fe31928f2ba06106fc2cd4282528b75d7917cba266a304785264945b58e`.
      Disabling AVX2 produces the same fingerprint in 51.749 seconds. The
      reviewed CPU fingerprint came from a former planner that admitted up to
      16 pixels of right padding; restoring that allowance on the current tree
      produces a third fingerprint, not the old one. Keep execution identities
      distinct, retain input-equivalent exact-width batching, and require
      independent text truth plus A3S Office reconstruction. Do not relax a
      golden or hide drift behind a corpus rule.
- [x] Trace the current 29-page CPU Text critical path without sample selectors.
      The 30.028-second run spends 22.858 wall seconds in recognition and 5.929
      in detection after bounded parallel execution of 242.253 and 10.627 summed
      CPU seconds respectively; crop and tensor preparation total about 1.017
      seconds. Spatial, terminal-projection, outer-window, and biased-activation
      one-variable isolations retain the same fingerprint, while serial windows
      regress to 74.554 seconds. Scheduling is not the remaining 10x lever.
- [ ] Add a lower-compute CPU profile only as a separate digest-pinned model or
      precision revision. Require independent text, geometry, table, seal,
      cross-page, and Office-reconstruction gates, including randomized order
      and out-of-corpus documents. Never select precision from file, page,
      content, source hash, provider/model label, or sample identity.
- [x] Reject implicit CPU F16/BF16 conversion. A temporary content-free
      four-shape probe measured F16 1.85--2.18x slower than F32 and found BF16
      matrix multiplication unsupported in the active backend. The diagnostic
      was removed; ISA presence alone does not change the declared model dtype.
- [x] Reject worker-occupancy pointwise scheduling after the real Text stage
      failed to improve. One content-free shape fell to 0.885 ms with bitwise
      parity, but the 29-page candidate took 31.071 seconds between retained
      30.670/30.874-second baselines. Add no batch-count or shape exception.
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
- [x] Pin Power's exact private contiguous
      `BatchNormalization -> erf-GELU` lowering. The reviewed batch-128 graph
      slice saves 182--192 microseconds and an alternating 29-page full-stage
      comparison reduces mean latency from 3,232.997 to 3,156.856 ms (2.36%)
      without changing text or semantic fingerprints.
- [x] Consume the exact rank-three classifier transpose view directly through
      Power's CUDA strided GEMM instead of materializing it. A content-free
      three-geometry gate is bit-exact and the recognition probe removes all
      386 matching transpose launches. Nine interleaved 29-page samples per
      binary reduce combined mean latency by 0.51% and median latency by 1.13%
      with identical 2,518-block text and semantic fingerprints; shared-GPU
      noise prevents a stable throughput claim, and the older strict semantic
      golden remains unchanged and open.
- [x] Lock exactly two source `Add`-`Identity`-`Sigmoid`-`Mul` last-axis
      biased-Swish windows and use Power's generic private CUDA F32 lowering.
      Generic unrelated-geometry parity is bit-exact and a launch-blocked trace
      removes 284 dynamic pointwise launches. Two independent precommitted
      interleaved cohorts improve separately; ten 29-page samples per binary
      reduce combined mean latency by 4.46%, median latency by 3.14%, and move
      mean throughput from 9.337 to 9.757 pages/s with identical 2,518-block
      text and semantic fingerprints. Stable 10-pages/s and the older strict
      semantic-golden gates remain open.
- [x] Lock nine source `MatMul -> Add` windows and let Power reuse the private
      CUDA F32 GEMM output for exact last-axis bias, composing the two internal
      retained Swish tails. The terminal classifier remains under its existing
      projection; the other eight windows remove eight allocation/free pairs
      without reducing kernel count. Generic rank-two through rank-four and
      storage-offset parity is byte-exact. Ten 29-page samples per binary
      improve combined mean latency 1.56%, but the reverse cohort improves only
      0.34% by mean and regresses 0.66% by median; the isolated-graph median also
      regresses. Retain as deterministic work removal with mixed evidence, not
      stable 10-pages/s proof.
- [x] Require exact leading-axis CUDA reduction partitions for recognition
      calls above 32 items. Power performs full spatial lowering once and
      writes each at-most-32 GEMM partition directly into one output. A
      batch-128 full-graph probe compares 95,795,200 values with zero differing
      bits. Keep OCR's production batch default at 32: the `64..256` complete-
      stage sweep was non-monotonic, all 41 tested explicit cuBLAS algorithms
      failed generic exact parity, and a four-lane follow-up regressed.
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
- [x] Make source-backed wired topology depend only on detector authority,
      local line contrast, connected junction geometry, exact centerline
      T-junction crossbars, and one bounded unique rectangular partition.
      Internal terminal clusters cannot invent axes, and only the nearest
      recurring terminal may close an open component side.
- [x] Retain 21/21 reviewed source grids with zero model fallback on pages
      7--24 and zero wired candidates on certificate-texture pages 26/28/29.
      On the Xeon w5-2445 CPU, the 29 decoded-raster/23-table source stage has a
      three-run median of 108.896 pages/s. Decoding, rasterization, text, seals,
      Parser, and Office reconstruction remain outside this measurement.
- [x] Add optional PicoDet-L model-backed seal positions with pinned assets,
      exact source-pixel geometry, confirmed versus boundary-candidate status,
      bounded immediate-predecessor edge views, and retained real rider-seal
      evidence for three interior marks plus two adjacent-page edge fragments.
- [x] Move exact decoder necessary conditions ahead of PicoDet local and edge
      admission. Full-page and achromatic adjacent-page paths remain intact;
      the retained 29-page positive gate falls from 67 to 44 model views and
      now completes in 24.255 CPU seconds (1.196 pages/s) while preserving all
      eight complete seals and four reconciled fragments. The 35-page precision
      gate completes in 21.286 seconds (1.644 pages/s) and retains only the
      reviewed invitation seal. These are seal-stage CPU results, not a
      complete fine-parse or CUDA throughput claim.
- [x] Derive the DocumentFast public model declaration from the exact admitted
      PicoDet weight SHA-256/size profile. PicoDet-S and PicoDet-L now retain
      distinct provider/result identities, and Parser retained-cache schema v4
      binds the exact live extractor identity. Ambiguous v2/v3 captures are not
      selected by the v4 real-corpus gate. The former hard-coded L declaration
      is retained as a failed provenance design in the canonical negative-result
      ledger.
- [x] Derive the DocumentFast text declaration from the typed PP-OCRv6
      detection/recognition profile. Freeze the exact Power session
      specification at composite-provider construction, validate lane outputs
      and batch receipts against it, and fail closed on later configuration
      mutation. Full-small and small-detection/tiny-recognition executions can
      no longer share one provider/result identity.
- [x] Let a complete confirmed seal dominate overlapping censored boundary
      geometry only from typed status, clipping, exact-profile NMS authority,
      smaller-region overlap, and complete-center containment. All 17 decoder
      tests preserve distinct objects. Current rider and precision gates retain
      eight confirmed plus two reconciled page-1/page-2 elements and exactly one
      invitation seal, with no content or sample selector.
- [x] Add opt-in pinned page orientation with typed quarter-turn transforms and
      transform-consistency abstention on the immutable source canvas.
- [ ] Add general layout, borderless-table, formula, and seal-text providers
      only with pinned assets, typed outputs, source-pixel geometry, and
      evidence.
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
      the current isolated-session full-stage median is 3.673 seconds (7.896
      pages/s), leaving about 0.773 seconds to the decoded-raster 10-pages/s
      gate. Complete PDF rasterization and Office reconstruction still require
      separate end-to-end evidence. The 2026-08-27 two-thread CPU trace is more
      constrained: six pages spend 13.101 seconds in 147 recognition graphs and
      2.322 seconds in detection, while table execution and composition total
      only about 13.7 milliseconds. Do not target the latter stages or silently
      substitute the faster tiny-recognition profile.
- [ ] Convert the 2026-08-25 exact-L all-corpus diagnostic into stable release
      evidence. The current unguarded 64-page decoded-raster sum is 6.231
      seconds (about 10.271 pages/s) and passes rider `10/10`, invitation `1/1`,
      merged-table `68/68`, cache replay, and Office reconstruction. The shared
      WDDM GPU was not continuously guarded and every reconstructed source still
      reports `fine_parse_ready=false`; repeat on an idle named host with
      p50/p95, memory, complete PDF rendering, and all open reconstruction gaps.
- [x] Validate Power's topology-only private constant-`Reshape` fold against
      frozen executables. Each current PicoDet-L call removes exactly 16 of 28
      executed reshapes and exposes existing convolution channel-bias fusion;
      current 64-page cache/reconstruction hashes, `68/68` table cells, and seal
      position/precision gates remain exact without model or sample selection.
- [x] Validate Power's lower-work private CUDA F32 sigmoid product on the
      official PicoDet-L graph. Retain the already materialized spatial gate,
      fuse only the full-shape `Sigmoid -> Mul`, and select from topology,
      liveness, dtype, device, contiguity, broadcast geometry, cancellation, and
      bounds. Four Mul execution boundaries disappear per graph call; complete
      CPU/CUDA graph suites, five normalized 64-page caches, and all 74 raw
      Office reconstruction artifacts remain exact.
- [x] Validate Power's adjacent private F32
      `BatchNormalization -> Sigmoid` output-pass lowering. Preserve Swish
      precedence, require one private normalized-value consumer, and do not
      extend or reorder convolution. The official layout graph removes four
      more execution boundaries per call; complete graph suites, all five
      normalized 64-page caches, and all 74 raw Office artifacts remain exact.
- [x] Bound dependency-independent PicoDet execution to four CUDA sessions
      from live device memory and typed batch demand. A strict fixed-order
      four/five-session cohort retained eight normalized-exact 64-page caches,
      but the fifth session won only 2/4 aggregate pairs and 1/4 pairs on the
      only treatment-active document; it regressed that document's mean and
      median and was removed. CPU and Metal remain single-session, and no
      source, page, text, pixel, hash, corpus, or observed-timing selector is
      present.
- [x] Restore exact concurrent cuBLAS behavior through Power's model-session
      workspaces and corrected handle/stream/buffer lifetime. Synchronize and
      reset the handle before freeing its workspace so independently retained
      devices cannot inherit a dangling address. No model, graph, source, page,
      content, geometry, corpus, or timing selector exists.
- [x] Reject and remove cross-branch refinement pooling. The strict eight-run
      static/pooling cohort retained exact normalized caches, but means were
      6,210.25/6,307 ms, medians were 6,116/6,316 ms, and pooling won one of four
      adjacent pairs. All layout phases again remain inside their static
      source/supplement partitions.
- [x] Rebuild the final static integration binary as
      `58763c3308f0fb8d50abf4decfa1f22d1e86ec4c27b8c9a1d23a60db61717aac`.
      Two fresh 64-page caches equal the strict control at normalized SHA-256
      `8df7df06a2a87950d233ad319a571b2eb07d0698e35a6972c1044fbad8c5ad7b`;
      all 74 Office artifacts remain path-and-byte identical.
- [ ] Complete the strict final workspace-lifetime performance non-regression
      cohort. The first admission launched zero binaries because unrelated
      compiler processes appeared; discard functional cache and process times.
- [ ] Complete the identical trace-free fixed-order performance cohort. The
      first comparison used unequal tracing and the first strict qualification
      launched zero samples after unrelated compilers appeared. Keep the older
      six-page text-golden failure open on both frozen binaries rather than
      updating it from current self-consistency.
- [ ] Compare the frozen constant-Reshape baseline with the combined lower-work
      sigmoid-product and BN/Sigmoid candidate under the same strict cohort.
      The combined correctness run emitted about 10.665 pages/s, but it was not
      an adjacent guarded comparison. Its following resource snapshot found
      active compilers and about 16% aggregate GPU utilization, so zero strict
      samples were admitted and no stable speed claim is available.

## Cross-repository sequence

1. A3S Power publishes the required model-neutral contract and TEE evidence.
2. A3S OCR pins it and implements model-specific batching and geometry.
3. A3S Parser pins the compatible OCR revision and owns render/OCR pipeline
   windows, persistence, reconciliation, cross-page graphs, and overlays.

An OCR optimization is not complete if it requires Parser to understand tensor
shapes or Power to understand OCR models.

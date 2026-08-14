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
- [x] Source-pixel polygons/boxes, provider/model fingerprints, and Power
      execution receipts.

### O1 — Staged image batches

- [x] Typed stage requests, stable slot IDs, exact cardinality/order, isolated
      failures, and completed/failed/skipped/unsupported outcomes.
- [x] Exact Power model sessions, finite queues, current-memory microbatch
      plans, one shared permit, cancellation, and receipt-v4 evidence.
- [x] Bounded batches of 1 through 256 source slots, detection microbatches of
      at most 16 images, and recognition batches of at most eight crops.

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
- [x] Cover mixed canvas shapes, padding exclusion, coordinate projection,
      output slicing, bounds, and an official-model scalar/batch gate with the
      TurboOCR-derived ASCII-token F1 floor of 0.95.
- [ ] Persist clean named-hardware scalar/batch reports before enabling a
      release-wide throughput claim.

### O3 — Cross-image recognition width buckets

- [x] Flatten detected crops across admitted images while retaining exact
      `(slot, detection, reading-order)` identity and materializing no more
      than one eight-crop recognition batch at a time.
- [x] Stable-sort and batch only identical dynamic canvas widths, preserving
      scalar recognition geometry while restoring results and shared receipts
      to original image and block order. Retain an isolated scalar retry for
      failed shared calls.
- [ ] Define measured static shape classes and a fill threshold with the exact
      dynamic-width path as fallback; never pad unboundedly merely to hit a
      static class.
- [ ] Prove CTC text/confidence parity, empty-image behavior, partial failures,
      cancellation, and scalar/batch receipt semantics on the official image
      matrix.

### O4 — Device-resident OCR stages

- [ ] Move resize/normalize and perspective ROI warp to reviewed OCR-owned
      device kernels when Power's bounded device-resident handles are ready.
- [ ] Evaluate DB threshold/connected-components and CTC argmax kernels behind
      exact CPU parity gates. Keep canonical CPU fallbacks explicit.
- [ ] Avoid full device-to-host tensor copies when only bounded maps, boxes, or
      token indices are needed.
- [ ] Retain Power admission, TEE/confidential-device policy, cancellation, and
      receipt binding for every fast path.

### O5 — OCR capability depth

- [ ] Add explicit orientation, layout, table, and formula providers/stages only
      with pinned assets, typed outputs, source-pixel geometry, and evidence.
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

## Cross-repository sequence

1. A3S Power publishes the required model-neutral contract and TEE evidence.
2. A3S OCR pins it and implements model-specific batching and geometry.
3. A3S Parser pins the compatible OCR revision and owns render/OCR pipeline
   windows, persistence, reconciliation, cross-page graphs, and overlays.

An OCR optimization is not complete if it requires Parser to understand tensor
shapes or Power to understand OCR models.

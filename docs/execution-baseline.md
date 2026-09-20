# PP-OCRv6 Execution Baseline Protocol

`a3s-use-ocr-execution-bench` records the real single-image A3S OCR baseline
used to evaluate batching, pooling, and admission changes. It executes the
public `OcrClient` path with the embedded PP-OCRv6 provider and A3S Power native
graph runtime. It does not invoke TurboOCR, Paddle, Python, ONNX Runtime, an OCR
service, or a subprocess.

The report fixes `evidenceScope` to `a3s-ocr-real-single-image` and
`providerClass` to `embedded-native`. It is not a Parser control-plane, Office
renderer, multi-surface throughput, accuracy, or cross-host performance claim.

## Fixed workload

The only accepted fixture is PaddleOCR's `general_ocr_002` object with:

- byte length `128713`;
- SHA-256
  `4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4`;
- decoded dimensions `896 x 528`; and
- detected media type `image/jpeg`.

The upstream URL and conventional local filename end in `.png`, but the pinned
bytes have a JPEG signature. The benchmark intentionally trusts the bytes, not
the extension. It rejects every other length, digest, media type, or decoded
dimension.

The existing official-image test remains the accuracy gate. It checks 30
ordered blocks against the pinned Paddle reference with reviewed text,
confidence, and polygon tolerances. The benchmark additionally requires every
sample to return those 30 blocks, eight schema-v1 A3S Power execution receipts,
no warnings, and byte-identical canonical output.

## Cold and warm sessions

One process owns one `OcrClient` and one lazily loaded `PpOcrV6Provider`:

1. The cold-start sample is the first extraction on that provider. Its measured
   interval includes public-client file read and hashing, image decode, model
   resolution, model and graph loading, Power weight verification, and all
   detection and recognition work. Provider construction, readiness diagnosis,
   and the benchmark's fixture verification occur before the interval.
2. Optional warmup extractions reuse the loaded engine and are validated but
   are not reported as measured samples.
3. Warm samples reuse the same loaded engine. They still include the public
   client file read, source hash, image decode, preprocessing, inference,
   postprocessing, validation, and result assembly.

“Cold start” means a cold model session, not a guaranteed cold operating-system
filesystem cache. A report must retain its build profile and source-tree state.
Only `release` reports from a `clean` exact revision are candidates for durable
performance evidence; `debug` or `modified` reports are diagnostic smoke data.

## Measurements and evidence

Each sample records:

- total elapsed nanoseconds;
- time to first result, equal to total elapsed time because the current public
  API publishes one atomic `OcrResult` rather than streaming internal blocks;
- integer milli-images per second derived from elapsed time;
- resident bytes before and after the call plus peak process resident bytes
  sampled every millisecond;
- block and Power receipt counts; and
- byte length and SHA-256 of canonical output evidence.

The canonical digest covers provider, engine, model, source media type/length/
SHA-256, recognized text, blocks, execution receipts, and warnings. It omits
the source path. The JSON report contains neither recognized text nor a model
directory or fixture path. It retains two sorted execution fingerprints for
detection and recognition: model family, revision, weight SHA-256, Power
runtime/version, and device.

Resident memory is process-wide, not allocator or tensor attribution. Linux
uses `VmRSS`, Windows uses `GetProcessMemoryInfo().WorkingSetSize`, and macOS
uses `getrusage(RUSAGE_SELF).ru_maxrss`. The sampler reports transient tensor
pressure as well as persistent model state. Compare reports only when fixture,
revision, profile, OS, architecture, CPU, RAM, Power device, sample procedure,
and runtime fingerprints are understood.

## Running a formal capture

First run `tools/check_official_ppocr_v6.sh` in its dedicated directory or
perform the equivalent pinned installation and official tests. Then use a
release build, a clean exact OCR revision, an honest stable host label, at least
one warmup, and enough measured samples for a useful p95:

```bash
export A3S_OCR_MODEL_DIR=/absolute/path/to/PP-OCRv6_small

cargo run --release --locked \
  --no-default-features \
  --features benchmark \
  --bin a3s-use-ocr-execution-bench -- \
  --ocr-commit 0123456789abcdef0123456789abcdef01234567 \
  --source-tree-state clean \
  --host-label a3s-lab-workstation-01 \
  --cpu-model "Named CPU" \
  --ram-bytes 137438953472 \
  --fixture /absolute/path/to/general_ocr_002.png \
  --warmup-samples 1 \
  --samples 10
```

The binary writes one self-validating JSON report to stdout. The schema accepts
smaller sample counts and `modified`/`debug` metadata for development smoke
runs, but those captures must not be promoted as release baselines.

Operator worktree diagnostics follow the same rule. A production fast path may
be selected from immutable device/ISA facts, declared operator topology, tensor
dtype/layout/geometry, and resource bounds, but not from a file name, page
number, source hash, recognized string, model label, corpus identity, or a
threshold fitted to observed fixtures. Compare the same semantic output with
the optimization enabled and disabled, retain exact or reviewed numerical
parity, and revert neutral or regressing variants instead of adding a
sample-specific rescue gate.

## Negative experiment retention

Every ineffective, regressing, mixed, inconclusive, reverted, parity-failing,
misconfigured, or zero-sample optimization and validation attempt must be
recorded before its source is removed or another hypothesis starts. The
canonical cross-repository history is the Parser
[append-before-removal negative-result ledger](https://github.com/contra-sense/agentic-parser/blob/main/docs/ocr-acceleration-plan.md#2026-08-24-onward-performance-negative-result-ledger).
It records the hypothesis, exact evidence boundary, parity result, local and
end-to-end measurements when available, missing evidence, decision, and
retained/reverted state. A later successful implementation does not erase the
failed attempts that led to it, and an invalid build or diagnostic command is
never silently converted into absent evidence.

Pure protocol tests do not require model assets:

```bash
cargo test --locked \
  --no-default-features \
  --features benchmark \
  --bin a3s-use-ocr-execution-bench
```

## Remaining TO1 evidence

This protocol closes the executable real-provider single-image measurement
slice. TO1 still requires clean release captures on the supported operating
systems, Power queue/residency observations when those public contracts exist,
and production A3S Office multi-surface render plus OCR evidence. Synthetic
Parser fixtures and this single-image OCR workload cannot substitute for those
claims.

The cross-image detection path additionally requires a batch report that runs
mixed aspect ratios through the public staged API, compares every slot with its
scalar result, records actual microbatch width and graph receipts, and measures
peak host/device memory. Until that clean named-hardware report is persisted,
the implementation is available without a release-wide throughput claim.
The official low-level gate also executes one pinned real-image crop at scalar
and cross-image batch width two. It requires exact text and source geometry,
recognition confidence within `0.00001`, a shared recognition receipt on both
slots, and an exact doubling of the receipt-bound input tensor size. This is a
numerical and mapping gate, not a substitute for the public multi-image report.
The checked-in official-model CI gate follows TurboOCR's accuracy contract: the
ASCII-token F1 between each mixed-shape batch slot and its scalar result must be
at least 0.95, while every polygon and box must remain inside its own source
image. Letterboxing changes convolution boundary context, so this is a bounded
quality gate rather than a claim of bit-identical detector tensors.
It also verifies both branches of the OCR-owned exact-work rule: a candidate
shares one Power-admitted detection graph call only when combined canvas-area
times cardinality does not exceed separate execution work, while any
work-increasing shape starts a distinct graph call with its own receipt.

Recognition batching has a stricter geometry rule. A CUDA diagnostic that
mixed different dynamic widths produced 0.933 ASCII-token F1 for the wide slot;
the same result occurred with the prior pinned Power revision. The checked path
therefore groups only exactly equal recognition tensor widths into one
canonical at-most-eight-crop cohort. Adjacent canonical cohorts share one
at-most-128-crop physical call only while that exact width remains equal, and
the input-plus-classifier reservation against Power's tensor limit derives a
smaller cap for wider tensors. No
empirical width delta or corpus-tuned padding threshold is admitted. The gate
must compare every
parallelized perspective crop and tensor slot with scalar materialization and
preserve exact input order and values. SHA-pinned Parser table and rider-seal
fixtures keep exact text and structured-geometry fingerprints under this
policy, along with their cell, IoU, and cross-page assertions. Detector inputs
use a 896-pixel fast bound and preserve original-source crops. Visually
non-uniform empty results receive one scalar retry at the 4,000-pixel quality
bound; this does not certify partially detected small text or replace the open
official-image matrix gate.

Fingerprint review must compare like execution identities. The historical CPU
rider golden was produced while the planner allowed up to 16 pixels of
recognition right padding. Re-enabling that allowance on the current binary
produced neither the historical CPU result nor the exact-width result. The old
golden is therefore not an operator-tuning oracle, and the current exact-width
fingerprint is not accepted by self-consistency alone. A replacement requires
independent transcription truth and A3S Office reconstruction, in addition to
the existing geometry, table, seal, and cross-page gates. No expected output may
be expanded merely because a faster candidate repeats it.

The current private-constant-`Reshape` Power experiment follows that rule. Its
frozen baseline and candidate reproduce the same current full-corpus evidence,
but the six-page table text fingerprint remains `d2329b...` against the older
`d675b5...` expectation on both binaries. Exact current parity exonerates the
executor rewrite; it does not certify the earlier text transition. The first
unequal-trace A/B and the later zero-sample quiet qualification are invalid
performance evidence and remain in Parser's append-before-removal ledger.

The following generic private `Sigmoid -> broadcast Mul` candidate is evaluated
against that retained constant-Reshape binary, not against a moving working
tree. Its frozen Parser executable has SHA-256
`62175a47d82eaa920ff2d57be42bbcdaa668268ef563e4277b872b6ab4683cfa`.
The official PicoDet-L trace removes four Mul execution boundaries per graph
call. All five 64-page cache wires match after replacing only their single
`elapsedMillis` value, and 74/74 A3S Office reconstruction artifacts match raw
bytes. These are correctness and activation gates only. The first resource
snapshot exceeded the declared 12% clean-GPU limit before either binary ran, so
no timing from the correctness runs enters the execution baseline.

The combined private `Sigmoid -> broadcast Mul` plus adjacent
`BatchNormalization -> Sigmoid` Parser executable is frozen separately with
SHA-256
`d4b8d347799bc17374c28ea7b94b244b03787ed1f6bfefcda43291907679e698`.
The additional lowering removes four more official-layout execution boundaries
per call. Complete graph suites, five normalized cache wires over 64 pages, and
all 74 raw Office reconstruction artifacts retain exact parity. Its emitted
6.001-second OCR sum (about 10.665 pages/s) is excluded from this baseline: it
was a correctness run, not an adjacent guarded comparison, and the following
strict preflight exceeded the same clean-GPU limit before either frozen binary
ran.

The scheduler-cap gate must alternate the previous and candidate caps on the
same named hardware. It requires exact text, block order, source geometry,
detection confidence, and every non-receipt canonical field. Recognition
confidence is reported separately as maximum absolute and ULP drift because
CUDA convolution arithmetic may vary with batch shape; that value must remain
finite, bounded to `[0,1]`, and must not control a runtime branch. The retained
RTX 4090 comparison for 32 versus 128 crops measured 6.889 versus 6.472 seconds
over 29 decoded pages and a maximum confidence difference of
`1.704692841e-5`.

Likewise, a reduced-precision or smaller-model benchmark must declare a distinct
model and execution fingerprint. It must run the same bounded, order-independent
corpus matrix and quality gates as its full-precision reference. Automatic
quantization or precision selection based on source, page, decoded content, or
sample identity is not admissible release evidence.

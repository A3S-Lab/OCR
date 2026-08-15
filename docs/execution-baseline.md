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
It also verifies both branches of the OCR-owned 90% canvas-fill rule: compatible
mixed shapes share one Power-admitted graph call, while a quality outlier starts
a distinct Power plan with its own receipt.

Recognition batching has a stricter geometry rule. A CUDA diagnostic that
mixed different dynamic widths produced 0.933 ASCII-token F1 for the wide slot;
the same result occurred with the prior pinned Power revision. The checked path
therefore never performs unbounded width mixing. It admits only crops whose
recognition canvases differ by at most 16 pixels into one at-most-eight-crop
call. Since the minimum canvas is 320 pixels, the maximum added right padding
is 5%; larger differences retain separate dynamic calls. SHA-pinned Parser
table and rider-seal fixtures keep exact text fingerprints under this bound,
along with their structured geometry and cross-page assertions. Detector
inputs use a 896-pixel fast bound and preserve original-source crops. Visually
non-uniform empty results receive one scalar retry at the 4,000-pixel quality
bound; this does not certify partially detected small text or replace the open
official-image matrix gate.

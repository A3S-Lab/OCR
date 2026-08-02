#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1}" || "${1}" == "/" ]]; then
  echo "usage: $0 <dedicated-test-model-directory>" >&2
  exit 2
fi

test_root="${1}"
if [[ -n "${HOME:-}" && "${test_root}" == "${HOME}" ]]; then
  echo "refusing to use the home directory as the PP-OCRv6 test root" >&2
  exit 2
fi
if [[ -e "${test_root}" ]]; then
  echo "the PP-OCRv6 test root must not already exist: ${test_root}" >&2
  exit 2
fi

mkdir -p "${test_root}"
test_root="$(cd "${test_root}" && pwd -P)"
export A3S_USE_OCR_HOME="${test_root}"

cargo run \
  --locked \
  --no-default-features \
  --features ppocr-v6 \
  --example install_models

export A3S_PPOCR_V6_MODEL="${test_root}/PP-OCRv6_small"
test -s "${A3S_PPOCR_V6_MODEL}/det/model.safetensors"
test -s "${A3S_PPOCR_V6_MODEL}/det/inference.yml"
test -s "${A3S_PPOCR_V6_MODEL}/rec/model.safetensors"
test -s "${A3S_PPOCR_V6_MODEL}/rec/inference.yml"

cargo test \
  --locked \
  --no-default-features \
  --features ppocr-v6 \
  --lib \
  ppocr_v6::native::tests::official_weights_execute_with_pinned_cpu_fixtures \
  -- \
  --ignored \
  --exact | tee "${test_root}/official-model-test.log"

grep -Fq \
  "test result: ok. 1 passed; 0 failed; 0 ignored" \
  "${test_root}/official-model-test.log"

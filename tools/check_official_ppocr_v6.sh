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

fixture_url="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
fixture_bytes="128713"
fixture_sha256="4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4"
export A3S_PPOCR_V6_REAL_IMAGE="${test_root}/general_ocr_002.png"

curl \
  --proto '=https' \
  --proto-redir '=https' \
  --fail \
  --location \
  --silent \
  --show-error \
  --max-filesize 1048576 \
  --output "${A3S_PPOCR_V6_REAL_IMAGE}" \
  "${fixture_url}"

actual_fixture_bytes="$(wc -c < "${A3S_PPOCR_V6_REAL_IMAGE}" | tr -d ' ')"
actual_fixture_sha256="$(shasum -a 256 "${A3S_PPOCR_V6_REAL_IMAGE}" | awk '{print $1}')"
test "${actual_fixture_bytes}" = "${fixture_bytes}"
test "${actual_fixture_sha256}" = "${fixture_sha256}"

cargo test \
  --locked \
  --no-default-features \
  --features ppocr-v6 \
  --lib \
  engine::tests::official_real_image_matches_upstream \
  -- \
  --ignored \
  --exact | tee "${test_root}/official-real-image-test.log"

grep -Fq \
  "test result: ok. 1 passed; 0 failed; 0 ignored" \
  "${test_root}/official-real-image-test.log"

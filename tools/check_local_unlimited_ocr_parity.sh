#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1}" || "${1}" == "/" ]]; then
  echo "usage: $0 <complete-reviewed-model-directory>" >&2
  exit 2
fi

model_root="$(cd "${1}" && pwd -P)"
if [[ -n "${HOME:-}" && "${model_root}" == "${HOME}" ]]; then
  echo "refusing to use the home directory as the Unlimited-OCR model root" >&2
  exit 2
fi

test_root="$(mktemp -d "${TMPDIR:-/tmp}/a3s-unlimited-parity.XXXXXX")"
cleanup() {
  rm -rf -- "${test_root}"
}
trap cleanup EXIT

export A3S_UNLIMITED_OCR_PARITY_IMAGE="${test_root}/general_ocr_002.png"
export A3S_UNLIMITED_OCR_MODEL_DIR="${model_root}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
"${script_root}/fetch_official_real_image.sh" "${A3S_UNLIMITED_OCR_PARITY_IMAGE}"

features=("unlimited-ocr")
if [[ "$(uname -s)" == "Darwin" ]]; then
  features=("unlimited-ocr-accelerate" "unlimited-ocr-metal")
fi

for feature in "${features[@]}"; do
  log_path="${test_root}/official-numerical-parity-${feature}.log"
  cargo test \
    --release \
    --locked \
    --no-default-features \
    --features "${feature}" \
    --lib \
    unlimited_ocr::numerical_tests::official_numerical_stability_and_grounding_match_upstream \
    -- \
    --ignored \
    --exact \
    --nocapture | tee "${log_path}"

  grep -Fq \
    "test result: ok. 1 passed; 0 failed; 0 ignored" \
    "${log_path}"
done

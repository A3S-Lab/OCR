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

export A3S_UNLIMITED_OCR_MODEL_DIR="${model_root}"
cargo test \
  --release \
  --locked \
  --no-default-features \
  --features unlimited-ocr \
  --lib \
  unlimited_ocr::assets::tests::complete_local_checkpoint_matches_reviewed_inventory \
  -- \
  --ignored \
  --exact

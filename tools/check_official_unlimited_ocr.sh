#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1}" || "${1}" == "/" ]]; then
  echo "usage: $0 <dedicated-test-fixture-directory>" >&2
  exit 2
fi

fixture_root="${1}"
if [[ -n "${HOME:-}" && "${fixture_root}" == "${HOME}" ]]; then
  echo "refusing to use the home directory as the Unlimited-OCR fixture root" >&2
  exit 2
fi
if [[ -e "${fixture_root}" ]]; then
  echo "the Unlimited-OCR fixture root must not already exist: ${fixture_root}" >&2
  exit 2
fi

mkdir -p "${fixture_root}"
fixture_root="$(cd "${fixture_root}" && pwd -P)"

repository="baidu/Unlimited-OCR"
revision="07dea832e22aefee32ad281d4b80551282e1c168"
weight_file="model-00001-of-000001.safetensors"
weight_bytes="6672547120"
weight_sha256="2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6"
header_json_bytes="334632"
base_url="https://huggingface.co/${repository}/resolve/${revision}"

download_asset() {
  local relative="${1}"
  local max_bytes="${2}"
  curl \
    --proto '=https' \
    --proto-redir '=https' \
    --fail \
    --location \
    --silent \
    --show-error \
    --retry 2 \
    --max-filesize "${max_bytes}" \
    --output "${fixture_root}/${relative}" \
    "${base_url}/${relative}"
}

download_asset "config.json" 131072
download_asset "tokenizer.json" 67108864
download_asset "tokenizer_config.json" 1048576
download_asset "special_tokens_map.json" 1048576
download_asset "processor_config.json" 1048576
download_asset "model.safetensors.index.json" 1048576

headers="${fixture_root}/model.safetensors.headers"
curl \
  --proto '=https' \
  --fail \
  --head \
  --silent \
  --show-error \
  --retry 2 \
  --output "${headers}" \
  "${base_url}/${weight_file}"

header_value() {
  local key="${1}"
  awk -F ': *' -v key="${key}" '
    tolower($1) == tolower(key) {
      sub(/\r$/, "", $2)
      print $2
      exit
    }
  ' "${headers}"
}

test "$(header_value x-repo-commit)" = "${revision}"
test "$(header_value x-linked-size)" = "${weight_bytes}"
test "$(header_value x-linked-etag)" = "\"${weight_sha256}\""

header_fixture="${fixture_root}/model.safetensors.header"
header_fixture_bytes="$((header_json_bytes + 8))"
curl \
  --proto '=https' \
  --proto-redir '=https' \
  --fail \
  --location \
  --silent \
  --show-error \
  --retry 2 \
  --range "0-$((header_fixture_bytes - 1))" \
  --max-filesize 1048576 \
  --output "${header_fixture}" \
  "${base_url}/${weight_file}"

actual_header_fixture_bytes="$(wc -c < "${header_fixture}" | tr -d ' ')"
actual_header_json_bytes="$(od -An -t u8 -N 8 "${header_fixture}" | tr -d '[:space:]')"
test "${actual_header_fixture_bytes}" = "${header_fixture_bytes}"
test "${actual_header_json_bytes}" = "${header_json_bytes}"

export A3S_UNLIMITED_OCR_OFFICIAL_FIXTURE="${fixture_root}"
cargo test \
  --locked \
  --no-default-features \
  --features unlimited-ocr \
  --lib \
  unlimited_ocr::assets::tests::official_checkpoint_header_matches_reviewed_inventory \
  -- \
  --ignored \
  --exact | tee "${fixture_root}/official-inventory-test.log"

grep -Fq \
  "test result: ok. 1 passed; 0 failed; 0 ignored" \
  "${fixture_root}/official-inventory-test.log"

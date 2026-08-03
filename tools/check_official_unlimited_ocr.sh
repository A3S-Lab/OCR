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
repository_url="https://huggingface.co/${repository}"
base_url="https://huggingface.co/${repository}/resolve/${revision}"

model_root="${fixture_root}/repository"
mkdir -p "${model_root}"
git -C "${model_root}" -c init.defaultBranch=main init --quiet
git -C "${model_root}" remote add origin "${repository_url}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${model_root}" fetch \
  --quiet \
  --depth=1 \
  origin \
  "${revision}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${model_root}" \
  -c advice.detachedHead=false \
  checkout --quiet --detach FETCH_HEAD

test "$(git -C "${model_root}" rev-parse HEAD)" = "${revision}"
test "$(wc -l < "${model_root}/${weight_file}" | tr -d ' ')" = "3"
test "$(sed -n '1p' "${model_root}/${weight_file}")" = \
  "version https://git-lfs.github.com/spec/v1"
test "$(sed -n '2p' "${model_root}/${weight_file}")" = \
  "oid sha256:${weight_sha256}"
test "$(sed -n '3p' "${model_root}/${weight_file}")" = \
  "size ${weight_bytes}"

header_fixture="${model_root}/model.safetensors.header"
header_fixture_bytes="$((header_json_bytes + 8))"
curl \
  --proto '=https' \
  --proto-redir '=https' \
  --fail \
  --location \
  --silent \
  --show-error \
  --retry 20 \
  --retry-all-errors \
  --retry-delay 15 \
  --retry-max-time 300 \
  --range "0-$((header_fixture_bytes - 1))" \
  --max-filesize 1048576 \
  --output "${header_fixture}" \
  "${base_url}/${weight_file}"

actual_header_fixture_bytes="$(wc -c < "${header_fixture}" | tr -d ' ')"
actual_header_json_bytes="$(od -An -t u8 -N 8 "${header_fixture}" | tr -d '[:space:]')"
test "${actual_header_fixture_bytes}" = "${header_fixture_bytes}"
test "${actual_header_json_bytes}" = "${header_json_bytes}"

export A3S_UNLIMITED_OCR_OFFICIAL_FIXTURE="${model_root}"
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

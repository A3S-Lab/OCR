#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "${1}" || "${1}" == "/" ]]; then
  echo "usage: $0 <new-output-file>" >&2
  exit 2
fi

output_path="${1}"
if [[ -e "${output_path}" ]]; then
  echo "the official real-image output must not already exist: ${output_path}" >&2
  exit 2
fi

fixture_url="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
fixture_bytes="128713"
fixture_sha256="4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4"

curl \
  --proto '=https' \
  --proto-redir '=https' \
  --fail \
  --location \
  --silent \
  --show-error \
  --max-filesize 1048576 \
  --output "${output_path}" \
  "${fixture_url}"

actual_fixture_bytes="$(wc -c < "${output_path}" | tr -d ' ')"
actual_fixture_sha256="$(shasum -a 256 "${output_path}" | awk '{print $1}')"
test "${actual_fixture_bytes}" = "${fixture_bytes}"
test "${actual_fixture_sha256}" = "${fixture_sha256}"

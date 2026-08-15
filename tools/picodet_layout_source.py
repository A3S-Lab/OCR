"""Pinned source identity for the reviewed PicoDet layout converter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


MODEL_NAME = "PicoDet-L_layout_3cls"
MODEL_JSON_SHA256 = "9df09659ed993444d068cc41b8b3e69306890b79c2af6f674d4111ab86e845da"
MODEL_PARAMS_SHA256 = "4baf2b29fdc3f8c4247f89b1126d267aa103463d1ed6e76068b073c8b0806c36"
MODEL_YAML_SHA256 = "f8aa3da98122157824ba5afad60b65aa00c5d530ce42ec861c570e1532f1376e"
MODEL_ARCHIVE_SHA256 = "a83d47f6bf27b14c593b8948b065d4779eda6ff8b3ab196e903853cdf69e2535"
INPUT_SIDE = 640
LOCATION_COUNT = 8_500
CLASS_COUNT = 3
RAW_WIDTH = 4 + CLASS_COUNT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source(root: Path) -> tuple[Path, Path, Path]:
    model = root / "inference.json"
    params = root / "inference.pdiparams"
    config = root / "inference.yml"
    expected = {
        model: MODEL_JSON_SHA256,
        params: MODEL_PARAMS_SHA256,
        config: MODEL_YAML_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"reviewed PicoDet source mismatch: {path}")
    metadata = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (
        metadata.get("Global", {}).get("model_name") != MODEL_NAME
        or metadata.get("label_list") != ["image", "table", "seal"]
        or metadata.get("Preprocess", [{}])[0].get("target_size")
        != [INPUT_SIDE, INPUT_SIDE]
    ):
        raise ValueError("reviewed PicoDet metadata changed")
    return model, params, config

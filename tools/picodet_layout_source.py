"""Pinned source identities for the reviewed PicoDet layout converter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ReviewedPicodetLayoutSource:
    """Exact upstream and runtime contract for one reviewed model artifact."""

    model_name: str
    family: str
    model_json_sha256: str
    model_params_sha256: str
    model_yaml_sha256: str
    model_archive_sha256: str
    input_side: int
    location_count: int
    class_count: int = 3

    @property
    def raw_width(self) -> int:
        return 4 + self.class_count


class PicodetLayoutProfile(str, Enum):
    LARGE = "large"
    SMALL = "small"


LARGE = ReviewedPicodetLayoutSource(
    model_name="PicoDet-L_layout_3cls",
    family="picodet-l-layout-3cls",
    model_json_sha256="9df09659ed993444d068cc41b8b3e69306890b79c2af6f674d4111ab86e845da",
    model_params_sha256="4baf2b29fdc3f8c4247f89b1126d267aa103463d1ed6e76068b073c8b0806c36",
    model_yaml_sha256="f8aa3da98122157824ba5afad60b65aa00c5d530ce42ec861c570e1532f1376e",
    model_archive_sha256="a83d47f6bf27b14c593b8948b065d4779eda6ff8b3ab196e903853cdf69e2535",
    input_side=640,
    location_count=8_500,
)

SMALL = ReviewedPicodetLayoutSource(
    model_name="PicoDet-S_layout_3cls",
    family="picodet-s-layout-3cls",
    model_json_sha256="d95d338030f9f8de79339fa7c4f99bac02255f48ee7af059cb60b09bd00188a6",
    model_params_sha256="6dfde7477ede8354d59854d05ca16ceeefaccc5f352c50653ec1d9381f855be2",
    model_yaml_sha256="b08eb43fd52e0b96bbeb3f82ccb607680a786d27c8b8ecfca7ea0221e8fb6f65",
    model_archive_sha256="8d9bd3ed048eb3f78a23eb1092471b00979bdaae16f607630dea36b521703bc9",
    input_side=480,
    location_count=4_789,
)

def source_for_profile(profile: PicodetLayoutProfile) -> ReviewedPicodetLayoutSource:
    if profile is PicodetLayoutProfile.LARGE:
        return LARGE
    if profile is PicodetLayoutProfile.SMALL:
        return SMALL
    raise ValueError(f"unreviewed PicoDet layout profile: {profile!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source(
    root: Path, source: ReviewedPicodetLayoutSource
) -> tuple[Path, Path, Path]:
    model = root / "inference.json"
    params = root / "inference.pdiparams"
    config = root / "inference.yml"
    expected = {
        model: source.model_json_sha256,
        params: source.model_params_sha256,
        config: source.model_yaml_sha256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"reviewed PicoDet source mismatch: {path}")
    metadata = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (
        metadata.get("Global", {}).get("model_name") != source.model_name
        or metadata.get("label_list") != ["image", "table", "seal"]
        or metadata.get("Preprocess", [{}])[0].get("target_size")
        != [source.input_side, source.input_side]
    ):
        raise ValueError("reviewed PicoDet metadata changed")
    return model, params, config

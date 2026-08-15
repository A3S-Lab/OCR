#!/usr/bin/env python3
"""Convert pinned OCR ONNX containers into native Power model assets.

ONNX is an offline interchange input only. The generated SafeTensors weights
and embedded A3S graph plans are the complete runtime inputs; a3s-power never
loads or executes ONNX.

Requires the development-only Python packages ``onnx``, ``numpy``, and
``safetensors``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


SCHEMA_VERSION = 1
SUPPORTED_OPS = {
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Conv",
    "ConvTranspose",
    "Div",
    "Erf",
    "GlobalAveragePool",
    "HardSigmoid",
    "Identity",
    "MatMul",
    "MaxPool",
    "Mul",
    "Pow",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}

SLANET_PLUS_ENCODER_SHA256 = (
    "dbd5431b4051b0f3037e3f8650dba4297cdf38a6a132ac9ccf57886184f4b66e"
)
SLANET_PLUS_SHAPE_CONSUMERS = {"Concat", "Slice"}
SLANET_PLUS_CONTROL_RESHAPE_OUTPUTS = {
    "helper.reshape.0",
    "helper.reshape.1",
    "helper.reshape.2",
    "helper.reshape.3",
    "helper.reshape.4",
    "helper.reshape.5",
    "helper.reshape.6",
}
SLANET_PLUS_RESIZE_SCALES = {
    "nearest_interp_v2_0.tmp_0": 31.0 / 16.0,
    "nearest_interp_v2_1.tmp_0": 61.0 / 31.0,
    "nearest_interp_v2_2.tmp_0": 2.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attribute_value(attribute: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, list)):
        return value
    raise ValueError(
        f"unsupported attribute {attribute.name!r} of type "
        f"{onnx.AttributeProto.AttributeType.Name(attribute.type)}"
    )


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_value:
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(dimension.dim_param)
        else:
            result.append("dynamic")
    return result


def convert(
    source: Path,
    role: str,
    output: Path,
    *,
    family: str = "pp-ocr-v6-small",
    slanet_plus_encoder: bool = False,
) -> None:
    source_sha256 = sha256(source)
    if slanet_plus_encoder and source_sha256 != SLANET_PLUS_ENCODER_SHA256:
        raise ValueError("SLANet-Plus lowering requires the reviewed encoder SHA-256")
    model = onnx.load(str(source), load_external_data=True)
    if slanet_plus_encoder:
        if (
            len(model.graph.input) != 1
            or model.graph.input[0].name != "x"
            or len(model.graph.output) != 1
            or model.graph.output[0].name != "transpose_0.tmp_0.0"
        ):
            raise ValueError("SLANet-Plus encoder I/O identity changed")
        # The reviewed split export omitted only the output ValueInfo shape.
        # Its fixed 488x488 path is independently parity-bound to [N,256,96].
        shape = model.graph.output[0].type.tensor_type.shape
        shape.ClearField("dim")
        shape.dim.add().dim_param = "batch"
        shape.dim.add().dim_value = 256
        shape.dim.add().dim_value = 96
    onnx.checker.check_model(model, full_check=True)
    lowered_ops = {"Cast", "HardSwish"} if slanet_plus_encoder else set()
    unsupported = sorted(
        {node.op_type for node in model.graph.node} - SUPPORTED_OPS - lowered_ops
    )
    if unsupported:
        raise ValueError(f"unsupported {family} operators: {unsupported}")

    tensors: dict[str, np.ndarray[Any, Any]] = {}
    initializers: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        value = np.ascontiguousarray(numpy_helper.to_array(initializer))
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype not in (np.float32, np.float16, np.int64, np.int32):
            raise ValueError(
                f"initializer {initializer.name!r} has unsupported dtype {value.dtype}"
            )
        tensors[initializer.name] = value
        initializers.append(
            {
                "name": initializer.name,
                "dtype": str(value.dtype),
                "shape": list(value.shape) or [1],
            }
        )

    consumers: dict[str, list[str]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node.op_type)

    nodes = []
    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}.{index}"
        inputs = [input_name for input_name in node.input if input_name]
        outputs = list(node.output)
        attributes = {
            attribute.name: attribute_value(attribute)
            for attribute in node.attribute
        }
        if slanet_plus_encoder and node.op_type == "HardSwish":
            if len(inputs) != 1 or len(outputs) != 1 or attributes:
                raise ValueError(f"unreviewed SLANet-Plus HardSwish node {name!r}")
            gate = f"{outputs[0]}.__hard_sigmoid"
            nodes.extend(
                [
                    {
                        "name": f"{name}.__hard_sigmoid",
                        "op": "HardSigmoid",
                        "inputs": inputs,
                        "outputs": [gate],
                        "attributes": {"alpha": 1.0 / 6.0, "beta": 0.5},
                    },
                    {
                        "name": f"{name}.__multiply",
                        "op": "Mul",
                        "inputs": [inputs[0], gate],
                        "outputs": outputs,
                        "attributes": {},
                    },
                ]
            )
            continue
        if slanet_plus_encoder and node.op_type == "Cast":
            if (
                len(inputs) != 1
                or len(outputs) != 1
                or attributes.get("to") not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64)
                or set(consumers.get(outputs[0], [])) - SLANET_PLUS_SHAPE_CONSUMERS
            ):
                raise ValueError(f"unreviewed SLANet-Plus shape Cast node {name!r}")
            nodes.append(
                {
                    "name": f"{name}.__shape_identity",
                    "op": "Identity",
                    "inputs": inputs,
                    "outputs": outputs,
                    "attributes": {},
                }
            )
            continue
        if (
            slanet_plus_encoder
            and node.op_type == "Reshape"
            and len(outputs) == 1
            and outputs[0] in SLANET_PLUS_CONTROL_RESHAPE_OUTPUTS
        ):
            if (
                len(inputs) != 2
                or attributes not in ({}, {"allowzero": 0})
                or consumers.get(outputs[0]) != ["Cast"]
            ):
                raise ValueError(
                    f"unreviewed SLANet-Plus control Reshape node {name!r}"
                )
            nodes.append(
                {
                    "name": f"{name}.__control_identity",
                    "op": "Identity",
                    "inputs": [inputs[0]],
                    "outputs": outputs,
                    "attributes": {},
                }
            )
            continue
        if (
            slanet_plus_encoder
            and node.op_type == "Resize"
            and len(outputs) == 1
            and outputs[0] in SLANET_PLUS_RESIZE_SCALES
        ):
            expected_attributes = {
                "coordinate_transformation_mode": "asymmetric",
                "mode": "nearest",
                "nearest_mode": "floor",
            }
            if len(inputs) != 4 or attributes != expected_attributes:
                raise ValueError(f"unreviewed SLANet-Plus Resize node {name!r}")
            scale_name = f"a3s.slanet_plus.resize_scales.{name}"
            scale = SLANET_PLUS_RESIZE_SCALES[outputs[0]]
            tensors[scale_name] = np.asarray([1.0, 1.0, scale, scale], dtype=np.float32)
            initializers.append(
                {
                    "name": scale_name,
                    "dtype": "float32",
                    "shape": [4],
                }
            )
            nodes.append(
                {
                    "name": name,
                    "op": "Resize",
                    "inputs": [inputs[0], inputs[1], scale_name],
                    "outputs": outputs,
                    "attributes": attributes,
                }
            )
            continue
        nodes.append(
            {
                "name": name,
                "op": node.op_type,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": attributes,
            }
        )

    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "family": family,
        "role": role,
        "source": {
            "format": "onnx",
            "sha256": source_sha256,
            "opset": max(entry.version for entry in model.opset_import),
        },
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.input
            if value.name not in tensors
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.output
        ],
        "initializers": sorted(initializers, key=lambda item: item["name"]),
        "nodes": nodes,
    }

    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"))
    (output / "graph.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", type=Path)
    parser.add_argument("--rec", type=Path)
    parser.add_argument("--table-encoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (arguments.det is None) != (arguments.rec is None):
        parser.error("--det and --rec must be supplied together")
    if arguments.det is None and arguments.table_encoder is None:
        parser.error("supply a PP-OCRv6 pair or --table-encoder")
    if arguments.det is not None:
        convert(arguments.det, "detection", arguments.output / "det")
        convert(arguments.rec, "recognition", arguments.output / "rec")
    if arguments.table_encoder is not None:
        convert(
            arguments.table_encoder,
            "table-encoder",
            arguments.output / "table",
            family="slanet-plus",
            slanet_plus_encoder=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack the reviewed Paddle 3 PicoDet layout graph for A3S Power.

Paddle is a development-only interchange reader. The generated SafeTensors
weights and graph plan are the complete runtime inputs; production A3S OCR
does not load Paddle, ONNX Runtime, Python, or an external inference service.

Requires ``paddlepaddle``, ``numpy``, ``pyyaml``, and ``safetensors``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import paddle
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picodet_layout_source import (
    CLASS_COUNT,
    INPUT_SIDE,
    LOCATION_COUNT,
    MODEL_ARCHIVE_SHA256,
    MODEL_JSON_SHA256,
    MODEL_NAME,
    MODEL_PARAMS_SHA256,
    MODEL_YAML_SHA256,
    RAW_WIDTH,
    require_source,
    sha256,
)


def operation_key(operation: Any) -> str:
    identifier = operation.id
    return str(identifier() if callable(identifier) else identifier)


def value_key(value: Any) -> str:
    identifier = value.id
    return str(identifier() if callable(identifier) else identifier)


def value_index(value: Any) -> int:
    index = value.index
    return int(index() if callable(index) else index)


class Converter:
    def __init__(self, root: Path) -> None:
        self.model, self.params, self.config = require_source(root)
        paddle.enable_static()
        self.executor = paddle.static.Executor(paddle.CPUPlace())
        self.program, feeds, _ = paddle.static.load_inference_model(
            str(root),
            self.executor,
            model_filename=self.model.name,
            params_filename=self.params.name,
        )
        if feeds != ["image", "scale_factor"]:
            raise ValueError(f"reviewed PicoDet feeds changed: {feeds}")
        self.operations = list(self.program.global_block().ops)
        self.operation_indices = {
            operation_key(operation): index
            for index, operation in enumerate(self.operations)
        }
        self.scope = paddle.static.global_scope()
        self.tensors: dict[str, np.ndarray[Any, Any]] = {}
        self.nodes: list[dict[str, Any]] = []
        self.reachable: set[str] = set()
        self.names: dict[str, str] = {}
        self.raw_boxes, self.raw_scores = self._raw_outputs()
        self._mark_value(self.raw_boxes)
        self._mark_value(self.raw_scores)

    def _raw_outputs(self) -> tuple[Any, Any]:
        nms = [
            operation
            for operation in self.operations
            if operation.name() == "pd_op.multiclass_nms3"
        ]
        if len(nms) != 1:
            raise ValueError("reviewed PicoDet graph must contain one NMS boundary")
        final_boxes, scores, _ = nms[0].operands_source()
        divide = final_boxes.get_defining_op()
        if divide.name() != "pd_op.divide":
            raise ValueError("reviewed PicoDet source scaling boundary changed")
        raw_boxes = divide.operands_source()[0]
        if list(raw_boxes.shape) != [-1, LOCATION_COUNT, 4]:
            raise ValueError("reviewed PicoDet raw box shape changed")
        if list(scores.shape) != [-1, CLASS_COUNT, LOCATION_COUNT]:
            raise ValueError("reviewed PicoDet raw score shape changed")
        return raw_boxes, scores

    def _mark_value(self, value: Any) -> None:
        if not value.initialized():
            return
        operation = value.get_defining_op()
        key = operation_key(operation)
        if key in self.reachable:
            return
        self.reachable.add(key)
        for source in operation.operands_source():
            self._mark_value(source)

    def _operation_name(self, operation: Any) -> str:
        index = self.operation_indices[operation_key(operation)]
        return f"paddle.{index}.{operation.name().replace('.', '_')}"

    def _value_name(self, value: Any) -> str:
        key = value_key(value)
        known = self.names.get(key)
        if known is not None:
            return known
        operation = value.get_defining_op()
        kind = operation.name()
        if kind == "builtin.parameter":
            name = str(operation.attrs()["parameter_name"])
        elif kind == "pd_op.data":
            name = str(operation.attrs()["name"])
        else:
            index = self.operation_indices[operation_key(operation)]
            name = f"paddle.value.{index}.{value_index(value)}"
        self.names[key] = name
        return name

    def _add_tensor(self, name: str, value: np.ndarray[Any, Any]) -> str:
        value = np.ascontiguousarray(value)
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype not in (np.float32, np.float16, np.int64, np.int32):
            raise ValueError(f"initializer {name!r} has unsupported dtype {value.dtype}")
        previous = self.tensors.get(name)
        if previous is not None and (
            previous.shape != value.shape or not np.array_equal(previous, value)
        ):
            raise ValueError(f"initializer {name!r} changed value")
        self.tensors[name] = value
        return name

    def _parameter(self, operation: Any) -> None:
        name = str(operation.attrs()["parameter_name"])
        variable = self.scope.find_var(name)
        if variable is None:
            raise ValueError(f"Paddle scope omitted parameter {name!r}")
        value = np.asarray(variable.get_tensor())
        self._add_tensor(name, value)

    def _constant(self, operation: Any) -> None:
        output = operation.results()[0]
        name = self._value_name(output)
        attributes = operation.attrs()
        if operation.name() == "pd_op.full_int_array":
            value = np.asarray(attributes["value"], dtype=np.int64)
        elif operation.name() == "pd_op.full":
            shape = [int(item) for item in attributes["shape"]]
            dtype = str(attributes["dtype"])
            normalized_dtype = dtype.lower()
            if "float32" in normalized_dtype:
                value = np.full(shape, float(attributes["value"]), dtype=np.float32)
            elif "int32" in normalized_dtype:
                value = np.full(shape, int(attributes["value"]), dtype=np.int32)
            else:
                raise ValueError(f"unsupported Paddle full dtype {dtype}")
        else:
            raise AssertionError("not a constant operation")
        self._add_tensor(name, value)

    def _scalar(self, value: Any) -> float:
        operation = value.get_defining_op()
        if operation.name() != "pd_op.full":
            raise ValueError("reviewed scalar control is not pd_op.full")
        return float(operation.attrs()["value"])

    def _combined_inputs(self, value: Any) -> list[str]:
        operation = value.get_defining_op()
        if operation.name() != "builtin.combine":
            raise ValueError("reviewed tensor list is not builtin.combine")
        return [self._value_name(item) for item in operation.operands_source()]

    def _node(
        self,
        operation: Any,
        op: str,
        inputs: list[str],
        output: str | None = None,
        attributes: dict[str, Any] | None = None,
        suffix: str = "",
    ) -> None:
        self.nodes.append(
            {
                "name": self._operation_name(operation) + suffix,
                "op": op,
                "inputs": inputs,
                "outputs": [output or self._value_name(operation.results()[0])],
                "attributes": attributes or {},
            }
        )

    def _conv(self, operation: Any) -> None:
        source, weight = operation.operands_source()
        weight_name = self._value_name(weight)
        shape = self.tensors[weight_name].shape
        attributes = operation.attrs()
        pads = [int(item) for item in attributes["paddings"]]
        self._node(
            operation,
            "Conv",
            [self._value_name(source), weight_name],
            attributes={
                "kernel_shape": list(shape[-2:]),
                "strides": list(attributes["strides"]),
                "pads": [pads[0], pads[1], pads[0], pads[1]],
                "dilations": list(attributes["dilations"]),
                "group": int(attributes["groups"]),
            },
        )

    def _batch_norm(self, operation: Any) -> None:
        source, mean, variance, scale, bias = operation.operands_source()
        self._node(
            operation,
            "BatchNormalization",
            [self._value_name(item) for item in (source, scale, bias, mean, variance)],
            attributes={"epsilon": float(operation.attrs()["epsilon"])},
        )

    def _hard_swish(self, operation: Any) -> None:
        source = self._value_name(operation.operands_source()[0])
        output = self._value_name(operation.results()[0])
        gate = output + ".hard_sigmoid"
        self._node(
            operation,
            "HardSigmoid",
            [source],
            output=gate,
            attributes={"alpha": 1.0 / 6.0, "beta": 0.5},
            suffix=".gate",
        )
        self._node(operation, "Mul", [source, gate], output=output, suffix=".multiply")

    def _scale(self, operation: Any) -> None:
        source, scale = operation.operands_source()
        output = self._value_name(operation.results()[0])
        bias = float(operation.attrs()["bias"])
        product = output if bias == 0.0 else output + ".scaled"
        self._node(
            operation,
            "Mul",
            [self._value_name(source), self._value_name(scale)],
            output=product,
            suffix=".multiply",
        )
        if bias != 0.0:
            bias_name = self._add_tensor(
                output + ".bias", np.asarray([bias], dtype=np.float32)
            )
            self._node(
                operation,
                "Add",
                [product, bias_name],
                output=output,
                suffix=".bias",
            )

    def _concat(self, operation: Any) -> None:
        combined, axis = operation.operands_source()
        self._node(
            operation,
            "Concat",
            self._combined_inputs(combined),
            attributes={"axis": int(self._scalar(axis))},
        )

    def _split(self, operation: Any) -> None:
        vector = operation.operands_source()[0].get_defining_op()
        if vector.name() != "pd_op.split_with_num":
            raise ValueError("reviewed split does not unpack split_with_num")
        source, axis_value = vector.operands_source()
        axis = int(self._scalar(axis_value))
        rank = len(source.shape)
        if axis < 0:
            axis += rank
        offset = 0
        for result in operation.results():
            if operation_key(operation) not in self.reachable:
                continue
            length = int(result.shape[axis])
            output = self._value_name(result)
            controls = []
            for label, values in (
                ("starts", [offset]),
                ("ends", [offset + length]),
                ("axes", [axis]),
                ("steps", [1]),
            ):
                controls.append(
                    self._add_tensor(
                        f"{output}.{label}", np.asarray(values, dtype=np.int64)
                    )
                )
            self._node(
                operation,
                "Slice",
                [self._value_name(source), *controls],
                output=output,
                suffix=f".{value_index(result)}",
            )
            offset += length

    def _resize(self, operation: Any) -> None:
        attributes = operation.attrs()
        if (
            list(attributes["scale"]) != [2.0, 2.0]
            or attributes["interp_method"] != "nearest"
            or attributes["align_corners"]
        ):
            raise ValueError("unreviewed PicoDet nearest interpolation policy")
        output = self._value_name(operation.results()[0])
        dummy = self._add_tensor(output + ".roi", np.asarray([0.0], dtype=np.float32))
        scales = self._add_tensor(
            output + ".scales", np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        )
        self._node(
            operation,
            "Resize",
            [self._value_name(operation.operands_source()[0]), dummy, scales],
            attributes={
                "coordinate_transformation_mode": "asymmetric",
                "mode": "nearest",
                "nearest_mode": "floor",
            },
        )

    def _pool(self, operation: Any) -> None:
        attributes = operation.attrs()
        output = operation.results()[0]
        if (
            attributes["pooling_type"] != "avg"
            or not attributes["adaptive"]
            or list(output.shape)[-2:] != [1, 1]
        ):
            raise ValueError("unreviewed PicoDet pooling policy")
        self._node(
            operation,
            "GlobalAveragePool",
            [self._value_name(operation.operands_source()[0])],
        )

    def _matmul(self, operation: Any) -> None:
        left, right = operation.operands_source()
        attributes = operation.attrs()
        if attributes["transpose_x"] or attributes["transpose_y"]:
            raise ValueError("unreviewed transposed PicoDet matmul")
        right_name = self._value_name(right)
        if self.tensors[right_name].ndim == 1:
            self.tensors[right_name] = np.ascontiguousarray(
                self.tensors[right_name].reshape((-1, 1))
            )
        self._node(
            operation,
            "MatMul",
            [self._value_name(left), right_name],
        )

    def _convert_operation(self, operation: Any) -> None:
        kind = operation.name()
        if kind == "builtin.parameter":
            self._parameter(operation)
        elif kind in ("pd_op.full", "pd_op.full_int_array"):
            self._constant(operation)
        elif kind in ("pd_op.data", "builtin.combine", "pd_op.split_with_num"):
            return
        elif kind in ("pd_op.conv2d", "pd_op.depthwise_conv2d"):
            self._conv(operation)
        elif kind == "pd_op.batch_norm_":
            self._batch_norm(operation)
        elif kind == "pd_op.hardswish":
            self._hard_swish(operation)
        elif kind == "pd_op.hardsigmoid":
            source = self._value_name(operation.operands_source()[0])
            attributes = operation.attrs()
            self._node(
                operation,
                "HardSigmoid",
                [source],
                attributes={
                    "alpha": float(attributes["slope"]),
                    "beta": float(attributes["offset"]),
                },
            )
        elif kind in ("pd_op.add", "pd_op.multiply", "pd_op.divide"):
            mapped = {"pd_op.add": "Add", "pd_op.multiply": "Mul", "pd_op.divide": "Div"}
            self._node(
                operation,
                mapped[kind],
                [self._value_name(item) for item in operation.operands_source()],
            )
        elif kind in ("pd_op.relu", "pd_op.sigmoid", "pd_op.sqrt"):
            mapped = {"pd_op.relu": "Relu", "pd_op.sigmoid": "Sigmoid", "pd_op.sqrt": "Sqrt"}
            self._node(
                operation,
                mapped[kind],
                [self._value_name(operation.operands_source()[0])],
            )
        elif kind == "pd_op.scale":
            self._scale(operation)
        elif kind == "pd_op.concat":
            self._concat(operation)
        elif kind == "builtin.split":
            self._split(operation)
        elif kind == "pd_op.nearest_interp":
            self._resize(operation)
        elif kind == "pd_op.pool2d":
            self._pool(operation)
        elif kind == "pd_op.reshape":
            self._node(
                operation,
                "Reshape",
                [self._value_name(item) for item in operation.operands_source()],
            )
        elif kind == "pd_op.transpose":
            self._node(
                operation,
                "Transpose",
                [self._value_name(operation.operands_source()[0])],
                attributes={"perm": list(operation.attrs()["perm"])},
            )
        elif kind == "pd_op.softmax":
            self._node(
                operation,
                "Softmax",
                [self._value_name(operation.operands_source()[0])],
                attributes={"axis": int(operation.attrs()["axis"])},
            )
        elif kind == "pd_op.matmul":
            self._matmul(operation)
        else:
            raise ValueError(f"unsupported reachable PicoDet operation: {kind}")

    def convert(self, output: Path) -> None:
        for operation in self.operations:
            if operation_key(operation) in self.reachable:
                self._convert_operation(operation)

        scores = self._value_name(self.raw_scores)
        boxes = self._value_name(self.raw_boxes)
        transposed_scores = "a3s.picodet.raw_scores"
        raw_output = "a3s.picodet.raw"
        self.nodes.extend(
            [
                {
                    "name": "a3s.picodet.transpose_scores",
                    "op": "Transpose",
                    "inputs": [scores],
                    "outputs": [transposed_scores],
                    "attributes": {"perm": [0, 2, 1]},
                },
                {
                    "name": "a3s.picodet.concat_raw",
                    "op": "Concat",
                    "inputs": [boxes, transposed_scores],
                    "outputs": [raw_output],
                    "attributes": {"axis": 2},
                },
            ]
        )
        initializers = [
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(self.tensors.items())
        ]
        plan = {
            "schemaVersion": 1,
            "family": "picodet-l-layout-3cls",
            "role": "layout-raw-head",
            "source": {
                "format": "paddle-pir",
                "sha256": MODEL_JSON_SHA256,
                "opset": 3,
            },
            "inputs": [{"name": "image", "shape": ["batch", 3, INPUT_SIDE, INPUT_SIDE]}],
            "outputs": [
                {
                    "name": raw_output,
                    "shape": ["batch", LOCATION_COUNT, RAW_WIDTH],
                }
            ],
            "initializers": initializers,
            "nodes": self.nodes,
        }
        output.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors, str(output / "model.safetensors"))
        (output / "graph.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "model": MODEL_NAME,
            "archiveSha256": MODEL_ARCHIVE_SHA256,
            "source": {
                "inference.json": MODEL_JSON_SHA256,
                "inference.pdiparams": MODEL_PARAMS_SHA256,
                "inference.yml": MODEL_YAML_SHA256,
            },
            "generated": {
                "graph.json": sha256(output / "graph.json"),
                "model.safetensors": sha256(output / "model.safetensors"),
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    Converter(arguments.source.resolve()).convert(arguments.output.resolve())


if __name__ == "__main__":
    main()

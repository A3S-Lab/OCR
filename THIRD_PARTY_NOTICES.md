# Third-Party Notices

## Baidu Unlimited-OCR

The optional `unlimited-ocr` feature implements the reviewed
`baidu/Unlimited-OCR` topology in native Rust and loads an operator-supplied
checkpoint at the pinned upstream revision. The crate does not redistribute
model weights, upstream Python code, vLLM, or container images. No upstream
source code is copied into the native implementation.

Unlimited-OCR is licensed under the MIT License.

Upstream repositories:

- <https://github.com/baidu/Unlimited-OCR>
- <https://huggingface.co/baidu/Unlimited-OCR>

## Pillow-Compatible Bicubic Preprocessing

The native Unlimited-OCR preprocessor independently implements the reviewed
8-bit bicubic coefficient, antialiasing, boundary, and fixed-point rounding
semantics used by Pillow. Numeric fixtures were derived from Pillow 11.3.0.
No Pillow runtime or Python dependency is included.

Pillow is licensed under the HPND License.

Reference: <https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c>

## PaddlePaddle/PaddleOCR PP-OCRv6 Models

A3S OCR release archives redistribute weights converted without numerical
changes from the official `PP-OCRv6_small_det` and `PP-OCRv6_small_rec`
inference bundles published by PaddlePaddle/PaddleOCR. ONNX is used only as the
audited offline interchange input to the OCR-owned conversion tool. Runtime
packages contain SafeTensors weights and inference configuration only. The
installer pins the A3S OCR release archive URL, byte size, SHA-256 digest, and
the canonical A3S Power weight digests.

PaddleOCR is licensed under the Apache License, Version 2.0.

Upstream repository: <https://github.com/PaddlePaddle/PaddleOCR>

Model collection: <https://huggingface.co/collections/PaddlePaddle/pp-ocrv6>

## A3S Power and Candle

Native OCR providers execute through the model-neutral `a3s-power` embedded
runtime. A3S Power uses the Candle Rust tensor library for embedded tensor
operations. Both projects are licensed under the MIT License or the Apache
License, Version 2.0, as applicable to their distributions.

Upstream repositories:

- <https://github.com/A3S-Lab/Power>
- <https://github.com/huggingface/candle>

## image-rs/imageproc

`a3s-use-ocr` uses `imageproc` version `0.25.0` for geometric image
transformations. `imageproc` is licensed under the MIT License.

Copyright (c) 2015 PistonDevelopers.

Upstream repository: <https://github.com/image-rs/imageproc>

## clipper2

`a3s-use-ocr` uses the `clipper2` Rust crate version `0.5.3` and
`clipper2c-sys` version `0.1.6` for bounded polygon offsetting during DB
post-processing. The Rust crates are available under the MIT License or the
Apache License, Version 2.0. Their bundled Clipper2 C/C++ implementation is
licensed under the Boost Software License, Version 1.0.

Upstream repositories:

- <https://github.com/tirithen/clipper2>
- <https://github.com/tirithen/clipper2c-sys>
- <https://github.com/AngusJohnson/Clipper2>

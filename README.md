### WORK-IN-PROGRESS

# PaddlePaddle to ONNX model convertor

Converts a PaddlePaddle model (`ProgramDesc` + parameters) into an ONNX graph. Uses the ONNX pip library and target PaddlePaddle **Fluid**.

To understand PaddlePaddle's (non-)graph way of representing a deep learning program, a `ProgramDesc`, refer to: https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md.

## Status

Targets Paddle->ONNX conversion for now, and will consequently support the reverse too.

Currently a work-in-progress tool since there a features in PaddlePaddle not supported in ONNX today and vice-versa.

## Usage

First, generate model directory by running any fluid test / example and write the model using the `fluid.io.save_inference_model` API.

Then, run `convert.py` by providing the generated model directory to the argument `---modeldir`.


## Installation

(TBD)

Create a virtual environment and install ONNX using PIP.
```
pip install onnx==1.1
```

Build PaddlePaddle's `develop` branch from source using info here:
http://paddlepaddle.org/docs/develop/documentation/en/build_and_install/build_from_source_en.html

## Testing

TBD

## Supported models

We aim to at least support all the models from our model bank. During our preliminary stage, we plan to support the models generated from:

- [fit_a_line](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)
- [fit_a_line](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)

## License
Provided under the [Apache-2.0 license](LICENSE).

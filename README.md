# PaddlePaddle to ONNX model convertor

Converts a PaddlePaddle model (`ProgramDesc` + parameters) into an ONNX model, with a goal to support inference of PaddlePaddle models across hardware platforms. Uses the ONNX pip helper library and targets models constructed with PaddlePaddle's **Fluid** API. Written in Python 2.7 (and underneath the hood, ONNX binds its Python helpers to their C++ packages).

# Usage

Before running the convertor,
- Install all the necessary dependencies (see "Installation" section below)
- Generate/save a PaddlePaddle Fluid model using the generate model directory by running any fluid test / example and write the model using the `fluid.io.save_inference_model` API (see [some of the examples](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/book) that you can plug this into).
  - A simple model for `fit_a_line` has been provided within the `extras` directory of this repo, to help you skip this step if you just wish to test it out.

Then, run the following:

```
python fluid_to_onnx.py --fluid_model <path_to_paddlepaddle_fluid.model> --onnx_model <path_to_where_you_want_to_output_model.onnx>
```

This should output an ONNX model (current version of opset: 6) which can be run on an ONNX backend for inference.


# How it works

- A (design document of the underlying ideas)[https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/onnx/onnx_convertor.md] behind how this program converts the model from PaddlePaddle to ONNX.
- Understand (PaddlePaddle's (non-)graph way of representing a deep learning program)[https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md], a `ProgramDesc`.


## Status

Targets Paddle->ONNX conversion for now, and will consequently support the reverse too.

Currently a work-in-progress tool since there a features in PaddlePaddle not supported in ONNX today and vice-versa.


## Installation

(TBD)

If you don't already have protobuf installed on your computer, install it from here: https://github.com/google/protobuf. On Mac, to get the development version, use `brew install protobuf`.

Create a virtual environment and install dependencies.
```
virtualenv venv
source venv/bin/activate
sh setup.sh
```

Build PaddlePaddle's `develop` branch from source using info here:
http://paddlepaddle.org/docs/develop/documentation/en/build_and_install/build_from_source_en.html. Make the `paddle/python` available in the execution environment's PYTHONPATH, or `pip install` the wheel after building the target `paddle_python`.

NOTE: Make sure your virtual environment has the new Protobuf used by this project and the `onnx` dependency, as Paddle installation may try to downgrade it.

## Testing / validation

TBD


## Supported models

We aim to at least support all the models from our model bank. During our preliminary stage, we have validated the inference model's conversion on following models:

- [fit_a_line](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)
- [recognize_digits](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recognize_digits.py)
- [VGG16 & ResNet50](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_image_classification.py)

## License
Provided under the [Apache-2.0 license](LICENSE).

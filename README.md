### WORK-IN-PROGRESS

# PaddlePaddle to ONNX model convertor

Converts a PaddlePaddle model (`ProgramDesc` + parameters) into an ONNX graph. Uses the ONNX pip library and targets PaddlePaddle **Fluid**. Built in Python 2.7 (and underneath the hood, ONNX does a Pybind to their C++ libraries).

To understand PaddlePaddle's (non-)graph way of representing a deep learning program, a `ProgramDesc`, refer to: https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md.

## Status

Targets Paddle->ONNX conversion for now, and will consequently support the reverse too.

Currently a work-in-progress tool since there a features in PaddlePaddle not supported in ONNX today and vice-versa.

## Usage

First, generate model directory by running any fluid test / example and write the model using the `fluid.io.save_inference_model` API.

Then, run `convert.py` by providing the generated model directory to the argument `---modeldir`.


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

## Testing

TBD

## Supported models

We aim to at least support all the models from our model bank. During our preliminary stage, we have validated the inference model's conversion on following models:

- [fit_a_line](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)
- [recognize_digits](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recognize_digits.py)
- [VGG16 & ResNet50](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_image_classification.py)
- [MobileNet](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/mobilenet.py)
- [SE_ResNeXt](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/se_resnext.py)

## License
Provided under the [Apache-2.0 license](LICENSE).

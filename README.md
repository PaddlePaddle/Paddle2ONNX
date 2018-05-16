# PaddlePaddle to ONNX model convertor

Converts a PaddlePaddle model (`ProgramDesc` + parameters) into an ONNX model, with a goal to support inference of PaddlePaddle models across hardware platforms. Uses the ONNX pip helper library and targets models constructed with PaddlePaddle's **Fluid** API. Written in Python 2.7 (and underneath the hood, ONNX binds its Python helpers to their C++ packages).

## Usage

Before running the convertor,
- Install all the necessary dependencies (see "Installation" section below)
- Generate/save a PaddlePaddle Fluid model using the generate model directory by running any fluid test / example and write the model using the `fluid.io.save_inference_model` API (see [some of the examples](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/book) that you can plug this into).
  - A simple model for `fit_a_line` has been provided within the `extras` directory of this repo, to help you skip this step if you just wish to test it out.

Then, run the following:

```
python fluid_to_onnx.py --fluid_model <path_to_paddlepaddle_fluid.model> --onnx_model <path_to_where_you_want_to_output_model.onnx>
```

This should output an ONNX model (current version of opset: 6) which can be run on an ONNX backend for inference.


## Installation

1. Clone the `paddle-onnx` repository from GitHub (in future, we plan to integrate it into the core PaddlePaddle `pip` and Docker distributable packages).
2. Install the external dependencies.
    1. Create a virtual environment to isolate dependencies (Optional).

        ```virtualenv venv```

    2. If you are running on Ubuntu, run `./setup.sh` and move to the next step.

        On other systems, you need to install [protobuf](https://github.com/google/protobuf) manually before installing the Python package dependencies. On Mac, to get the development version, use `brew install protobuf`.

    3. If you created a virtual environment, activate it using:

        ```source venv/bin/activate```

3. We now need to make sure that we have a built PaddlePaddle package available for us. If you have this setup during your model building process, and have the `PYTHONPATH` of PaddlePaddle already set in the environment (this should be setup if you followed the right instructions or used Docker), you are all set and continue to use the converter!

    The way you can test this is by opening up the Python shell by running `python` on your main shell. And then, if you can `import paddle`, you are all set.

    If not, [build PaddlePaddle's `develop` branch from source](http://paddlepaddle.org/docs/develop/documentation/en/build_and_install/build_from_source_en.html). Make the `paddle/python` available in the execution environment's PYTHONPATH, or `pip install` the wheel after building the target `paddle_python`.

    *NOTE*: Make sure your virtual environment has the new Protobuf and the correct `onnx` dependency used by this project (see the version in the `requirements.txt` file), as the PaddlePaddle installation may try to downgrade it without asking you.


## How it works

- A [design document of the underlying ideas](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/onnx/onnx_convertor.md) behind how this program converts the model from PaddlePaddle to ONNX.
- Understand [PaddlePaddle's (non-)graph way of representing a deep learning program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md), a `ProgramDesc`.


## Status

The current release targets Paddle->ONNX conversion for now (or what's called **frontend** in the ONNX world), and will consequently support the reverse too.

Currently aimed at a wider coverage of models and operators. There are several PaddlePaddle model features not available in ONNX and some vice-versa, so will also aim to resolve these in near future. See the design document in the `How it works` section above for more details.




## Testing / validation

To validate the similarity of the PaddlePaddle Fluid model and the exported ONNX model, run the following:

```
python validate.py --fluid_model <path_to_paddlepaddle_fluid.model> --onnx_model <path_to_exported_model.onnx>
```

This validation aims for an output tensor comparison precision at 5-decimal places. To discover the other arguments to validation, run:

```
python validate.py --help
```


## Supported models

We aim to at least support all the models from our model bank. During our preliminary stage, we have validated the inference model's conversion on following models:

- [fit_a_line](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)
- [recognize_digits](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recognize_digits.py)
- [VGG16 & ResNet50](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_image_classification.py)
- [MobileNet](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/mobilenet.py)
- [SE_ResNeXt](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/se_resnext.py)
- [Inception-v4](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/inception_v4.py)


## Got feedback or issues?

If you have suggestions or bugs you run into, we invite you to share them in the **Issues** section of this repo. The maintainers of this project look at these issues everyday - you know, while squinting their eyes staring at their cell phones in the dark right after waking up.


## Contribute

In the highest spirits of open-source, we welcome your contributions! The philosophy around contribution at this stage is to get **better model coverage**. Some of the code for training popular models is already written for PaddlePaddle Fluid. For code not written for PaddlePaddle Fluid, we invite writing this training code first (and possibly contributing it to the `tests` inside PaddlePaddle's main repo) and then:

- Read the documents in the `How it works` section above.
- (Optional) Create an issue sharing need for support for a new model. Assign to yourself.
- Add the necessary operator conversion logic in `fluid_onnx/ops.py`.
- Write tests for the newly introduced operator convertor functions.
- Add the model to the list of supported models in this root `README.md` file.
- Open a new PR for the model(s) that resolves your issue. In your tests, post the output of your validation process.


## License
Provided under the [Apache-2.0 license](LICENSE).

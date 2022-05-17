#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import setuptools

long_description = "paddle2onnx is a toolkit for converting trained model of PaddlePaddle to ONNX.\n\n"
long_description += "Usage: paddle2onnx --model_dir src --save_file dist\n"
long_description += "GitHub: https://github.com/PaddlePaddle/paddle2onnx\n"
long_description += "Email: dltp-sz@baidu.com"

setuptools.setup(
    name="paddle2onnx",
    version="0.9.6",
    author="dltp-sz",
    author_email="dltp-sz@baidu.com",
    description="a toolkit for converting trained model of PaddlePaddle to ONNX.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/paddle2onnx",
    packages=setuptools.find_packages(),
    install_requires=['six', 'protobuf', 'onnx<=1.9.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['paddle2onnx=paddle2onnx.command:main']})

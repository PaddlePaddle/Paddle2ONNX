#!/bin/bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Detect the operating system
OS=$(uname -s)
ARCH=$(uname -m)

# Check if the operating system is Linux
if [ "$OS" = "Linux" ]; then
    if [[ "$ARCH" == "x86_64" ]]; then
      protobuf_tgz_name="protobuf-linux-x64-3.21.12.tgz"
    elif [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
      protobuf_tgz_name="protobuf-linux-aarch64-3.16.0.tgz"
    else
        echo "When the operating system is Linux, the system architecture only supports (x86_64 and aarch64), but the current architecture is $ARCH."
        exit 1
    fi
    protobuf_url="https://bj.bcebos.com/paddle2onnx/third_party/$protobuf_tgz_name"
# Check if the operating system is Darwin (macOS)
elif [ "$OS" = "Darwin" ]; then
    if [[ "$ARCH" == "x86_64" ]]; then
      protobuf_tgz_name="protobuf-osx-x86_64-3.16.0.tgz"
    elif [[ "$ARCH" == "arm64" ]]; then
      protobuf_tgz_name="protobuf-osx-arm64-3.16.0.tgz"
    else
      echo "When the operating system is Darwin, the system architecture only supports (x86_64 and arm64), but the current architecture is $ARCH."
      exit 1
    fi
    protobuf_url="https://bj.bcebos.com/fastdeploy/third_libs/$protobuf_tgz_name"
else
   echo "The system only supports (Linux and Darwin), but the current system is $OS."
fi

wget $protobuf_url
protobuf_save_dir="$PWD/installed_protobuf"
mkdir -p $protobuf_save_dir
tar -zxf $protobuf_tgz_name -C $protobuf_save_dir
export PATH=$protobuf_save_dir/bin:${PATH}

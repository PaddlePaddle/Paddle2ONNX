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

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

set -e -x

# CLI arguments
PY_VERSION=$1
PLAT=$2
SYSTEM_NAME=$3

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# Use system protobuf (much faster than compiling from source)
yum install -y protobuf-compiler protobuf-devel cmake python3-pip 2>/dev/null || \
  dnf install -y protobuf-compiler protobuf-devel cmake python3-pip 2>/dev/null || true

# Compile wheels
# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.9"]="cp39-cp39" ["3.10"]="cp310-cp310" ["3.11"]="cp311-cp311" ["3.12"]="cp312-cp312")
PY_VER=${python_map[$PY_VERSION]}
PIP_INSTALL_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir -q"
PYTHON_COMMAND="/opt/python/${PY_VER}/bin/python"

# Update pip and install build deps
$PIP_INSTALL_COMMAND --upgrade pip
$PIP_INSTALL_COMMAND cmake pybind11

# Build Paddle2ONNX wheels
$PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }

# Bundle external shared libraries into the wheels
# libpaddle.so is no longer required — PIR parsing is now standalone.
find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> /tmp/failed-wheels; }" \;

if [[ -f /tmp/failed-wheels ]]; then
    echo "Repairing wheels failed:"
    cat /tmp/failed-wheels
    exit 1
fi

# Remove useless *-linux*.whl; only keep manylinux*.whl
rm -f dist/*-linux*.whl

echo "Successfully build wheels:"
find . -type f -iname "*manylinux*.whl"

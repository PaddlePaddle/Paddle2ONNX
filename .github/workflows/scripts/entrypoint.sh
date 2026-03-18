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

# Compile wheels
# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39" ["3.10"]="cp310-cp310" ["3.11"]="cp311-cp311" ["3.12"]="cp312-cp312")
PY_VER=${python_map[$PY_VERSION]}
PIP_INSTALL_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir -q"
PYTHON_COMMAND="/opt/python/${PY_VER}/bin/python"

# Update pip and install cmake
$PIP_INSTALL_COMMAND --upgrade pip
$PIP_INSTALL_COMMAND cmake

# Build and install protobuf
original_dir=$(pwd)
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v21.12
git submodule update --init
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=`pwd`/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
export PATH=`pwd`/installed_protobuf_lib/bin:${PATH}
cd $original_dir

export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"

# Build Paddle2ONNX wheels
$PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }

#  ============================================================================
#   Due to libpaddle’s limitations, it can only maintain a restricted platform tag of ‘linux_x86_64’.
#  ============================================================================
# Bundle external shared libraries into the wheels
# find -exec does not preserve failed exit codes, so use an output file for failures
failed_wheels=$PWD/failed-wheels
rm -f "$failed_wheels"
find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --exclude libpaddle.so || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> '$failed_wheels'; }" \;

if [[ -f "$failed_wheels" ]]; then
    echo "Repairing wheels failed:"
    cat failed-wheels
    exit 1
fi

# Remove useless *-linux*.whl; only keep manylinux*.whl
rm -f dist/*-linux*.whl

echo "Successfully build wheels:"
find . -type f -iname "*manylinux*.whl"

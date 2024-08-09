#!/bin/bash

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

set -e -x

# CLI arguments
PY_VERSION=$1
PLAT=$2
SYSTEM_NAME=$3

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/lib64:/
ln -s /usr/lib64/libpython3.11.so.1.0 /usr/lib64/libpython3.11.so
# Compile wheels
# Need to be updated if there is a new Python Version
# declare -A python_map=( ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39" ["3.10"]="cp310-cp310" ["3.11"]="cp311-cp311" ["3.12"]="cp312-cp312")
# PY_VER=${python_map[$PY_VERSION]}
# PIP_INSTALL_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir -q"
PYTHON_COMMAND="/usr/local/bin/python3.11"

# Update pip and install cmake
$PYTHON_COMMAND -m pip install cmake
$PYTHON_COMMAND -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
# Build protobuf from source
if [[ "$SYSTEM_NAME" == "CentOS" ]]; then
    yum install -y wget
fi
source .github/workflows/scripts/download_protobuf.sh
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
# Build Paddle2ONNX wheels
$PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }
# Bundle external shared libraries into the wheels
# find -exec does not preserve failed exit codes, so use an output file for failures
# failed_wheels=$PWD/failed-wheels
# rm -f "$failed_wheels"
# find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}' || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> '$failed_wheels'; }" \;

# if [[ -f "$failed_wheels" ]]; then
#     echo "Repairing wheels failed:"
#     cat failed-wheels
#     exit 1
# fi

# Remove useless *-linux*.whl; only keep manylinux*.whl
rm -f dist/*-linux*.whl

echo "Successfully build wheels:"
find . -type f -iname "*manylinux*.whl"

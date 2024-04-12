#!/bin/bash

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

export CORE_NUMBER=$1

if [[ -z "$CORE_NUMBER" ]]; then
   export CORE_NUMBER=1
fi

# Build protobuf from source with -fPIC on Unix-like system
ORIGINAL_PATH=$(pwd)
cd ..
git clone https://github.com/protocolbuffers/protobuf.git -b v3.16.0
cd protobuf
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j$CORE_NUMBER && make install
export PATH=$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$INSTALL_PROTOBUF_PATH/bin:$PATH
cd $ORIGINAL_PATH
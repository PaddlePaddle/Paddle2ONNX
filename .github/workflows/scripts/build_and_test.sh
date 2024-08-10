#!/bin/bash

#build protobuf
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init
git checkout v4.22.0
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j200
make install
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}

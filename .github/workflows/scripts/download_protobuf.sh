#!/bin/bash

# Detect the operating system
OS=$(uname -s)
ARCH=$(uname -m)

# Check if the operating system is Linux
if [ "$OS" = "Linux" ]; then
    if [[ "$ARCH" == "x86_64" ]]; then
      protobuf_tgz_name="protobuf-linux-x64-3.16.0.tgz"
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
protobuf_svae_dir="$PWD/installed_protobuf"
mkdir -p $protobuf_svae_dir
tar -zxf $protobuf_tgz_name -C $protobuf_svae_dir
export PATH=$protobuf_svae_dir/bin:${PATH}
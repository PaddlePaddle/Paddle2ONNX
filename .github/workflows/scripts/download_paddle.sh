#!/bin/bash

# Detect the operating system
OS=$(uname -s)
ARCH=$(uname -m)

# Check if the operating system is Linux
if [ "$OS" = "Linux" ]; then
    if [[ "$ARCH" == "x86_64" ]]; then
      paddle_tgz_name="paddle-linux-x64-dev.tgz"
    elif [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
      paddle_tgz_name="paddle-linux-aarch64-dev.tgz"
    else
        echo "When the operating system is Linux, the system architecture only supports (x86_64 and aarch64), but the current architecture is $ARCH."
        exit 1
    fi
    paddle_url="https://paddle2onnx.bj.bcebos.com/third_libs//$paddle_tgz_name"
# Check if the operating system is Darwin (macOS)
elif [ "$OS" = "Darwin" ]; then
    if [[ "$ARCH" == "x86_64" ]]; then
      paddle_tgz_name="paddle-osx-x86_64-dev.tgz"
    elif [[ "$ARCH" == "arm64" ]]; then
      paddle_tgz_name="paddle-osx-arm64-dev.tgz"
    else
      echo "When the operating system is Darwin, the system architecture only supports (x86_64 and arm64), but the current architecture is $ARCH."
      exit 1
    fi
    paddle_url="https://paddle2onnx.bj.bcebos.com/third_libs/$paddle_tgz_name"
else
   echo "The system only supports (Linux and Darwin), but the current system is $OS."
fi

wget $paddle_url
paddle_svae_dir="$PWD/installed_paddle"
mkdir -p $paddle_svae_dir
tar -zxf $paddle_tgz_name -C $paddle_svae_dir
export PATH=$paddle_svae_dir/libs:${PATH}

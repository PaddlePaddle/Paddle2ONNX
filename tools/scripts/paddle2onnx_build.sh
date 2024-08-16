#!/bin/bash
set -x
set -e
PADDLE2ONNX_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

function download_protobuf(){
    OS=$(uname -s)
    ARCH=$(uname -m)

    # Check if the operating system is Linux
    if [ "$OS" = "Linux" ]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            protobuf_tgz_name="protobuf-linux-x64-3.21.12.tgz"
        elif [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
            protobuf_tgz_name="protobuf-linux-aarch64-3.21.12.tgz"
        else
            echo "When the operating system is Linux, the system architecture only supports (x86_64 and aarch64), but the current architecture is $ARCH."
            exit 1
        fi
        protobuf_url="https://bj.bcebos.com/paddle2onnx/third_party/$protobuf_tgz_name"
    # Check if the operating system is Darwin (macOS)
    elif [ "$OS" = "Darwin" ]; then
        if [[ "$ARCH" == "x86_64" ]]; then
        protobuf_tgz_name="protobuf-osx-x86_64-3.21.12.tgz"
        elif [[ "$ARCH" == "arm64" ]]; then
        protobuf_tgz_name="protobuf-osx-arm64-3.21.12.tgz"
        else
        echo "When the operating system is Darwin, the system architecture only supports (x86_64 and arm64), but the current architecture is $ARCH."
        exit 1
        fi
        protobuf_url="https://bj.bcebos.com/fastdeploy/third_libs/$protobuf_tgz_name"
    else
    echo "The system only supports (Linux and Darwin), but the current system is $OS."
    fi

    wget -q $protobuf_url
    protobuf_save_dir="$PWD/installed_protobuf"
    mkdir -p $protobuf_save_dir
    tar -zxf $protobuf_tgz_name -C $protobuf_save_dir
    export PATH=$protobuf_save_dir/bin:$protobuf_save_dir/lib64:${PATH}
}

function build_protobuf(){
    cd ${PADDLE2ONNX_ROOT}/protobuf
    mkdir build_source && cd build_source
    cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j200
    make install
    export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}
}
function build_paddle2onnx(){
    #download protobuf
    build_protobuf

    #install dependencies needed by building paddle2onnx
    $1 -m pip install --upgrade pip
    $1 -m pip install build
    $1 -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

    #build paddle2onnx
    export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
    cd ${PADDLE2ONNX_ROOT}
    $1 -m build --wheel
    $1 -m pip install dist/*.whl

}

function run_onnx_test() {
    cd ${PADDLE2ONNX_ROOT}/tests
    bash run.sh $1
}

function main() {
    local CMD=$1
    PY_VERSION=$2
    case $CMD in
      build_and_test_on_linux_x86)
        build_paddle2onnx ${PY_VERSION}
        run_onnx_test ${PY_VERSION}
        ;;
      *)
        echo "You did not enter a correct case!"
        exit 1
        ;;
      esac
}

main $@

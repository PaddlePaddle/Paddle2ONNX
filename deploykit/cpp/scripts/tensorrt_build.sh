#!/bin/bash

for i in "$@"; do
    case $i in
        --tensorrt_dir=*)
         TENSORRT_DIR="${i#*=}"
         shift
         ;;
        --tensorrt_header=*)
         TENSORRT_HEADER="${i#*=}"
         shift
         ;;
        --cuda_dir=*)
         CUDA_DIR="${i#*=}"
         shift
         ;;
        *)
         # unknown option
         exit 1
         ;;
    esac
done

if [ $TENSORRT_DIR ];then
	echo "TENSORRT_DIR = $TENSORRT_DIR"
else
	echo "TENSORRT_DIR is not exist, please set by --tensorrt_dir"
    exit 1
fi

if [ $CUDA_DIR ];then
	echo "CUDA_DIR = $CUDA_DIR"
else
	echo "CUDA_DIR is not exist, please set by --cuda_dir"
    exit 1
fi

if [ $TENSORRT_HEADER ];then
	echo " TENSORRT_HEADER= $TENSORRT_HEADER"
else
	echo "TENSORRT_HEADER is not exist, please set by --tensorrt_header"
    exit 1
fi
# download opencv library
OPENCV_DIR=$(pwd)/deps/opencv3.4.6gcc4.8ffmpeg/
{
    bash $(pwd)/scripts/bootstrap.sh ${OPENCV_DIR} # 下载预编译版本的加密工具和opencv依赖库
} || {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
}

# download glog library
GLOG_DIR=$(pwd)/deps/glog/
GLOG_URL=https://bj.bcebos.com/paddlex/deploy/glog.tar.gz

if [ ! -d $(pwd)deps/ ]; then
    mkdir -p deps
fi

if [ ! -d ${GLOG_DIR} ]; then
    cd deps
    wget -c ${GLOG_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz 
    rm -rf glog.tar.gz 
    cd ..
fi

# download gflogs library
GFLAGS_DIR=$(pwd)/deps/gflags/
GFLAGS_URL=https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz
if [ ! -d ${GFLAGS_DIR} ]; then
    cd deps
    wget -c ${GFLAGS_URL} -O glog.tar.gz
    tar -zxvf glog.tar.gz 
    rm -rf glog.tar.gz 
    cd ..
fi

rm -rf build
mkdir -p build
cd build
cmake ../demo/tensorrt_inference/ \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DTENSORRT_HEADER=${TENSORRT_HEADER} \
    -DCUDA_DIR=${CUDA_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR}  \
    -DGLOG_DIR=${GLOG_DIR} \
    -DGFLAGS_DIR=${GFLAGS_DIR}

make -j16


#!/bin/bash

set -e -x

PYTHON_COMMAND="/usr/bin/python3.8"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
# Update pip and install cmake
$PYTHON_COMMAND -m pip install --upgrade pip
$PYTHON_COMMAND -m pip install cmake
$PYTHON_COMMAND -m pip install build
$PYTHON_COMMAND -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# Build protobuf from source
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v4.22.0
git submodule update --init
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j30
make install

# 将编译目录加入环境变量
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}

cd ../..
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
# Build Paddle2ONNX wheels
$PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }

# Install Paddle2ONNX wheels
$PYTHON_COMMAND -m pip install dist/*.whl

#Run tests
cases=$(find ./tests/ -name "test*.py" | sort)
ignore="test_auto_scan_multiclass_nms.py
        test_auto_scan_roi_align.py \ # need to be rewrite
        test_auto_scan_pool_adaptive_max_ops.py \
        test_auto_scan_isx_ops.py \
        test_auto_scan_masked_select.py \
        test_auto_scan_pad2d.py \
        test_auto_scan_roll.py \
        test_auto_scan_set_value.py \
        test_auto_scan_unfold.py \
        test_auto_scan_uniform_random_batch_size_like.py \
        test_auto_scan_uniform_random.py \
        test_auto_scan_dist.py \
        test_auto_scan_distribute_fpn_proposals1.py \
        test_auto_scan_distribute_fpn_proposals_v2.py \
        test_auto_scan_fill_constant_batch_size_like.py \
        test_auto_scan_generate_proposals.py \
        test_uniform.py \
        test_ceil.py \
        test_deform_conv2d.py \
        test_floor_divide.py \
        test_has_nan.py \
        test_isfinite.py \
        test_isinf.py \
        test_isnan.py \
        test_mask_select.py \
        test_median.py \
        test_nn_Conv3DTranspose.py \
        test_nn_GroupNorm.py \
        test_nn_InstanceNorm3D.py \
        test_nn_Upsample.py \
        test_normalize.py \
        test_scatter_nd_add.py \
        test_unsqueeze.py \
        test_quantize_model.py \
        test_quantize_model_minist.py \
        test_quantize_model_speedup.py \
        test_resnet_fp16.py"
bug=0

# Install Python Packet
$PYTHON_COMMAND -m pip install pytest
$PYTHON_COMMAND -m pip install onnx onnxruntime tqdm filelock
$PYTHON_COMMAND -m pip install six hypothesis

echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        $PYTHON_COMMAND -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: ${bug}" >> result.txt
cat result.txt
exit "${bug}"

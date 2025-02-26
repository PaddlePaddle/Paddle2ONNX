# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

# TODO: Restore CI detection for quantization operators
#wget -P ~/.cache/paddle/dataset/int8/download/ http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model.tar.gz
#tar xf ~/.cache/paddle/dataset/int8/download/mnist_model.tar.gz -C ~/.cache/paddle/dataset/int8/download/
#wget -P ~/.cache/paddle/dataset/int8/download/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
#tar xf ~/.cache/paddle/dataset/int8/download/MobileNetV1_infer.tar -C ~/.cache/paddle/dataset/int8/download/
#wget -P ~/.cache/paddle/dataset/int8/download/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
#tar xf ~/.cache/paddle/dataset/int8/download/ResNet50_infer.tar -C ~/.cache/paddle/dataset/int8/download/
#wget -P ~/.cache/paddle/dataset/int8/download/ http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz
#mkdir ~/.cache/paddle/dataset/int8/download/small_data/ && tar xf ~/.cache/paddle/dataset/int8/download/calibration_test_data.tar.gz -C ~/.cache/paddle/dataset/int8/download/small_data/
#wget https://bj.bcebos.com/paddle2onnx/tests/quantized_models.tar.gz
#tar xf quantized_models.tar.gz

cases=$(find . -name "test*.py" | sort)
ignore="test_auto_scan_multiclass_nms.py
        test_auto_scan_roi_align.py \ # need to be rewrite
        test_auto_scan_pool_adaptive_max_ops.py \
        test_auto_scan_pad2d.py \
        test_auto_scan_unfold.py \
        test_auto_scan_uniform_random_batch_size_like.py \
        test_auto_scan_uniform_random.py \
        test_auto_scan_gaussian_random.py \
        test_auto_scan_distribute_fpn_proposals1.py \
        test_auto_scan_distribute_fpn_proposals_v2.py \
        test_auto_scan_fill_constant_batch_size_like.py \
        test_auto_scan_unary_ops.py \
        test_auto_scan_generate_proposals.py \
        test_uniform.py \
        test_deform_conv2d.py \
        test_hardtanh.py \
        test_nn_GRU.py \
        test_quantize_model.py \
        test_quantize_model_minist.py \
        test_auto_scan_partial_ops.py \
        test_dygraph2onnx.py \
        test_auto_scan_dequantize_linear.py \
        test_auto_scan_quantize_linear.py \
        test_quantize_model_speedup.py \
        test_resnet_fp16.py \
        test_empty.py \
        test_auto_scan_pool_max_ops.py \
        test_auto_scan_fill_constant.py"
bug=0

# Install Python Packet
export PY_CMD=$1
$PY_CMD -m pip install pytest
$PY_CMD -m pip install onnx onnxruntime tqdm filelock
$PY_CMD -m pip install six hypothesis
$PY_CMD -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

export ENABLE_DEV=ON
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        FLAGS_enable_pir_api=0 $PY_CMD -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: ${bug}" >> result.txt
cat result.txt
exit "${bug}"

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

cases=`find . -name "test*.py" | sort`
ignore="test_expand_as.py \
        test_split.py \
        test_uniform.py"
bug=0
export PY_CMD=$1
$PY_CMD -m pip install pytest

export ENABLE_DEV=OFF
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        $PY_CMD -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

export ENABLE_DEV=ON
dev_tests=("test_auto_scan_conv2d.py \
            test_auto_scan_unary_ops.py \
            test_auto_scan_assign.py \
            test_auto_scan_batch_norm.py \
            test_auto_scan_interpolate_ops.py \
            test_auto_scan_bmm.py \
            test_auto_scan_cast.py \
            test_auto_scan_clip.py \
            test_auto_scan_concat.py \
            test_auto_scan_conv2d.py \
            test_auto_scan_cumsum.py \
            test_auto_scan_conv2d.py \
            test_auto_scan_dropout.py \
            test_auto_scan_elementwise_ops.py \
            test_auto_scan_elu.py \
            test_auto_scan_expand_v2.py \
            test_auto_scan_fill_constant.py \
            test_auto_scan_fill_like.py \
            test_auto_scan_flatten.py \
            test_auto_scan_gather.py \
            test_auto_scan_gaussian_random.py \
            test_auto_scan_gelu.py \
            test_auto_scan_hardsigmoid.py \
            test_auto_scan_hardswish.py \
            test_auto_scan_layer_norm.py \
            test_auto_scan_leakyrelu.py \
            test_auto_scan_logsoftmax.py \
            test_auto_scan_logsigmoid.py \
            test_auto_scan_logsumexp.py \
            test_auto_scan_lookup_table_v2.py \
            test_auto_scan_matmul.py \
            test_auto_scan_matmul_v2.py \
            test_auto_scan_mean.py \
            test_auto_scan_meshgrid.py \
            test_auto_scan_logical_ops.py \
            test_auto_scan_pad3d.py \
            test_auto_scan_prelu.py \
            test_auto_scan_range.py \
            test_auto_scan_reducemean_ops.py \
            test_auto_scan_reshape.py \
            test_auto_scan_rnn.py \
            test_auto_scan_scale.py \
            test_auto_scan_shape.py \
            test_auto_scan_size.py \
            test_auto_scan_slice.py \
            test_auto_scan_softmax.py \
            test_auto_scan_shrink_ops.py \
            test_auto_scan_split.py \
            test_auto_scan_squeeze2.py \
            test_auto_scan_stack.py \
            test_auto_scan_unstack.py \
            test_auto_scan_strided_slice.py \
            test_auto_scan_sum.py \
            test_auto_scan_shrink_ops.py \
            test_auto_scan_thresholded_relu.py \
            test_auto_scan_tile.py \
            test_auto_scan_top_k_v2.py \
            test_auto_scan_transpose.py \
            test_auto_scan_unsqueeze2.py \
            test_auto_scan_where.py \
            test_auto_scan_yolo_box.py \
            test_auto_scan_dist.py \
            test_auto_scan_yolo_box.py \
            test_auto_scan_argsort.py \
            test_auto_scan_expand_as.py \
            test_auto_scan_log.py")
echo "=============dev test=========" >>result.txt
echo "=============dev test========="
for file in ${dev_tests}
do
    echo ${file}
    $PY_CMD -m pytest ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
    fi
done

echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}

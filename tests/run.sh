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
        test_auto_scan_softmax_with_cross_entropy.py \
        test_auto_scan_pool_adaptive_max_ops.py \
        test_auto_scan_top_k.py \
        test_auto_scan_flip.py \
        test_auto_scan_group_norm.py \
        test_auto_scan_index_select.py \
        test_auto_scan_instance_norm.py \
        test_auto_scan_interpolate_v1_ops.py \
        test_auto_scan_isx_ops.py \
        test_auto_scan_linspace.py \
        test_auto_scan_masked_select.py \
        test_auto_scan_mv.py \
        test_auto_scan_norm.py \
        test_auto_scan_pad2d.py \
        test_auto_scan_pixel_shuffle.py \
        test_auto_scan_p_norm.py \
        test_auto_scan_roll.py \
        test_auto_scan_scatter.py \
        test_auto_scan_set_value.py \
        test_auto_scan_top_k.py \
        test_auto_scan_unfold.py \
        test_auto_scan_uniform_random_batch_size_like.py \
        test_auto_scan_uniform_random.py \
        test_auto_scan_unique.py \
        test_auto_scan_dist.py \
        test_uniform.py \
        test_ceil.py \
        test_floor_divide.py \
        test_has_nan.py \
        test_index_select.py \
        test_isfinite.py \
        test_isinf.py \
        test_isnan.py \
        test_mask_select.py \
        test_median.py \
        test_mv.py \
        test_nn_AdaptiveAvgPool3D.py \
        test_nn_Conv3D.py \
        test_nn_Conv3DTranspose.py \
        test_nn_GroupNorm.py \
        test_nn_InstanceNorm3D.py \
        test_nn_MaxPool3D.py \
        test_nn_PixelShuffle.py \
        test_nn_Upsample.py \
        test_normalize.py \
        test_scatter_nd_add.py \
        test_scatter.py \
        test_unique.py \
        test_unsqueeze.py"
bug=0
export PY_CMD=$1
$PY_CMD -m pip install pytest

export ENABLE_DEV=ON
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

echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}

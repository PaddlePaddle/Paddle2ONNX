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

cases=$(find . -name "test*.py" | sort)
ignore="test_auto_scan_multiclass_nms.py
        test_auto_scan_generate_proposals.py
        test_quantize_model.py
        test_quantize_model_minist.py
        test_quantize_model_speedup.py
        test_dygraph2onnx.py
        test_resnet_fp16.py"
bug=0

# Install Python dependencies
export PY_CMD=$1
$PY_CMD -m pip install pytest onnx onnxruntime hypothesis
$PY_CMD -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

export ENABLE_DEV=ON
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "skipping ${file##*/}"
    else
        $PY_CMD -m pytest ${file}
        if [ $? -ne 0 ]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done

echo "total bugs: ${bug}" >> result.txt
cat result.txt
exit "${bug}"

@echo off
REM Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
REM Licensed under the Apache License, Version 2.0 (the "License")
REM You may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM     http://www.apache.org/licenses/LICENSE-2.0
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM Uncomment the lines below if you need to download and extract the datasets
REM wget -P "%USERPROFILE%\.cache\paddle\dataset\int8\download\" http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model.tar.gz
REM tar xf "%USERPROFILE%\.cache\paddle\dataset\int8\download\mnist_model.tar.gz" -C "%USERPROFILE%\.cache\paddle\dataset\int8\download\"
REM wget -P "%USERPROFILE%\.cache\paddle\dataset\int8\download\" https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
REM tar xf "%USERPROFILE%\.cache\paddle\dataset\int8\download\MobileNetV1_infer.tar" -C "%USERPROFILE%\.cache\paddle\dataset\int8\download\"
REM wget -P "%USERPROFILE%\.cache\paddle\dataset\int8\download\" https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
REM tar xf "%USERPROFILE%\.cache\paddle\dataset\int8\download\ResNet50_infer.tar" -C "%USERPROFILE%\.cache\paddle\dataset\int8\download\"
REM wget -P "%USERPROFILE%\.cache\paddle\dataset\int8\download\" http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz
REM mkdir "%USERPROFILE%\.cache\paddle\dataset\int8\download\small_data\" && tar xf "%USERPROFILE%\.cache\paddle\dataset\int8\download\calibration_test_data.tar.gz" -C "%USERPROFILE%\.cache\paddle\dataset\int8\download\small_data\"
REM wget https://bj.bcebos.com/paddle2onnx/tests/quantized_models.tar.gz
REM tar xf quantized_models.tar.gz

REM Find test files and prepare ignore list
setlocal enabledelayedexpansion

REM Replace this with actual files found in your environment or use a command like 'dir /s /b test*.py'
for /R %%i in (test*.py) do (
    set cases=!cases! %%i
)

REM List of files to ignore
set ignore=test_auto_scan_multiclass_nms.py
set ignore=!ignore! test_auto_scan_roi_align.py
set ignore=!ignore! test_auto_scan_pool_adaptive_max_ops.py
set ignore=!ignore! test_auto_scan_pad2d.py
set ignore=!ignore! test_auto_scan_unfold.py
set ignore=!ignore! test_auto_scan_uniform_random_batch_size_like.py
set ignore=!ignore! test_auto_scan_uniform_random.py
set ignore=!ignore! test_auto_scan_distribute_fpn_proposals1.py
set ignore=!ignore! test_auto_scan_distribute_fpn_proposals_v2.py
set ignore=!ignore! test_auto_scan_fill_constant_batch_size_like.py
set ignore=!ignore! test_auto_scan_generate_proposals.py
set ignore=!ignore! test_uniform.py
set ignore=!ignore! test_deform_conv2d.py
set ignore=!ignore! test_has_nan.py
set ignore=!ignore! test_unsqueeze.py
set ignore=!ignore! test_quantize_model.py
set ignore=!ignore! test_quantize_model_minist.py
set ignore=!ignore! test_quantize_model_speedup.py
set ignore=!ignore! test_resnet_fp16.py
set ignore=!ignore! test_empty.py
set ignore=!ignore! test_auto_scan_pool_max_ops.py
set ignore=!ignore! test_auto_scan_fill_constant.py
set ignore=!ignore! test_auto_scan_layer_norm.py
set ignore=!ignore! test_auto_scan_scatter_nd_add.py
REM uncomment below tests when using not paddlepaddle-gpu
set ignore=!ignore! test_auto_scan_assign.py
set ignore=!ignore! test_auto_scan_scatter_nd_add.py
set ignore=!ignore! test_auto_scan_conv2d.py
set ignore=!ignore! test_auto_scan_conv2d_transpose.py
set ignore=!ignore! test_auto_scan_conv3d.py
set ignore=!ignore! test_auto_scan_grid_sampler.py
set ignore=!ignore! test_auto_scan_dequantize_linear.py
set ignore=!ignore! test_auto_scan_gaussian_random.py
set ignore=!ignore! test_auto_scan_partial_ops.py
set ignore=!ignore! test_auto_scan_unary_ops.py

REM Initialize bug count
set bug=0

REM Install Python packages
set PY_CMD=%1
%PY_CMD% -m pip install pytest
%PY_CMD% -m pip install onnx onnxruntime tqdm filelock
%PY_CMD% -m pip install six hypothesis
REM %PY_CMD% -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
%PY_CMD% -m pip install https://paddle2onnx.bj.bcebos.com/paddle_windows/paddlepaddle_gpu-0.0.0-cp310-cp310-win_amd64.whl

REM Enable development mode and run tests
set FLAGS_enable_pir_api=0
set ENABLE_DEV=ON
echo ============ failed cases ============ >> result.txt

for %%f in (!cases!) do (
    echo %%f
    echo !ignore! | findstr /C:"%%~nxf" > nul
    if !errorlevel! equ 0 (
        echo Skipping %%f
    ) else (
        %PY_CMD% -m pytest %%f -s
        if !errorlevel! neq 0 (
            echo %%f >> result.txt
            set /a bug+=1
        )
    )
)

echo total bugs: !bug! >> result.txt
type result.txt

exit /b !bug!

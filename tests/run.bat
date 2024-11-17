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
set ignore=!ignore! test_auto_scan_isx_ops.py
set ignore=!ignore! test_auto_scan_masked_select.py
set ignore=!ignore! test_auto_scan_pad2d.py
set ignore=!ignore! test_auto_scan_roll.py
set ignore=!ignore! test_auto_scan_set_value.py
set ignore=!ignore! test_auto_scan_unfold.py
set ignore=!ignore! test_auto_scan_uniform_random_batch_size_like.py
set ignore=!ignore! test_auto_scan_uniform_random.py
set ignore=!ignore! test_auto_scan_dist.py
set ignore=!ignore! test_auto_scan_distribute_fpn_proposals1.py
set ignore=!ignore! test_auto_scan_distribute_fpn_proposals_v2.py
set ignore=!ignore! test_auto_scan_fill_constant_batch_size_like.py
set ignore=!ignore! test_auto_scan_generate_proposals.py
set ignore=!ignore! test_uniform.py
set ignore=!ignore! test_ceil.py
set ignore=!ignore! test_deform_conv2d.py
set ignore=!ignore! test_floor_divide.py
set ignore=!ignore! test_has_nan.py
set ignore=!ignore! test_isfinite.py
set ignore=!ignore! test_isinf.py
set ignore=!ignore! test_isnan.py
set ignore=!ignore! test_mask_select.py
set ignore=!ignore! test_median.py
set ignore=!ignore! test_nn_Conv3DTranspose.py
set ignore=!ignore! test_nn_GroupNorm.py
set ignore=!ignore! test_nn_InstanceNorm3D.py
set ignore=!ignore! test_nn_Upsample.py
set ignore=!ignore! test_normalize.py
set ignore=!ignore! test_scatter_nd_add.py
set ignore=!ignore! test_unsqueeze.py
set ignore=!ignore! test_quantize_model.py
set ignore=!ignore! test_quantize_model_minist.py
set ignore=!ignore! test_quantize_model_speedup.py
set ignore=!ignore! test_resnet_fp16.py

REM Initialize bug count
set bug=0

REM Install Python packages
set PY_CMD=%1
%PY_CMD% -m pip install pytest
%PY_CMD% -m pip install onnx onnxruntime tqdm filelock
%PY_CMD% -m pip install paddlepaddle==2.6.0
%PY_CMD% -m pip install six hypothesis

REM Enable development mode and run tests
set ENABLE_DEV=ON
echo ============ failed cases ============ >> result.txt

for %%f in (!cases!) do (
    echo %%f
    echo !ignore! | findstr /C:"%%~nxf" > nul
    if !errorlevel! equ 0 (
        echo Skipping %%f
    ) else (
        %PY_CMD% -m pytest %%f
        if !errorlevel! neq 0 (
            echo %%f >> result.txt
            set /a bug+=1
        )
    )
)

echo total bugs: !bug! >> result.txt
type result.txt

exit /b !bug!
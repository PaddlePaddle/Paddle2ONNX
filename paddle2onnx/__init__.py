# Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
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
import sys
import importlib.metadata
import packaging.version as pv

try:
    err_msg = (
        "Please install the latest paddle: python -m pip install --pre "
        "paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/, "
        "more information: https://www.paddlepaddle.org.cn/install/quick?docurl=undefined"
    )
    import paddle

    lib_paddle_name = (
        "paddlepaddle-gpu" if paddle.is_compiled_with_cuda() else "paddlepaddle"
    )
    paddle_version = importlib.metadata.version(lib_paddle_name)
    min_version = "3.0.0.dev20250426"
    if (
        sys.platform == "win32"
        and (
            pv.parse(paddle_version) < pv.parse(min_version)
            or paddle_version == "3.0.0"
        )
    ) or pv.parse(paddle_version) < pv.parse(min_version):
        raise ValueError(
            f"The paddlepaddle version should not be less than {min_version}. {err_msg}"
        )
except ImportError:
    raise ImportError(
        f"Failed to import paddle. Please ensure paddle is installed. {err_msg}"
    )

from .version import version
from .convert import export  # noqa: F401
from .convert import dygraph2onnx  # noqa: F401
from .convert import load_parameter  # noqa: F401
from .convert import save_program  # noqa: F401

__version__ = version

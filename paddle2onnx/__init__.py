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

import os
import sys
import ctypes
import glob

if sys.platform == "win32":
    pfiles_path = os.getenv("ProgramFiles", "C:\\Program Files")
    py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
    package_dir = os.path.dirname(os.path.abspath(__file__))
    libs_path = os.path.join(package_dir, "libs")

    if sys.exec_prefix != sys.base_exec_prefix:
        base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
    else:
        base_py_dll_path = ""

    dll_paths = list(filter(os.path.exists, [libs_path]))

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if with_load_library_flags:
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    dlls = None
    for dll_path in dll_paths:
        os.add_dll_directory(dll_path)
        dlls = glob.glob(os.path.join(dll_path, "*.dll"))

    try:
        ctypes.CDLL("vcruntime140.dll")
        ctypes.CDLL("msvcp140.dll")
        ctypes.CDLL("vcruntime140_1.dll")
    except OSError:
        print(
            """Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe"""
        )

    # Not load 32 bit dlls in 64 bit python.
    dlls = [dll for dll in dlls if "32_" not in dll]
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if with_load_library_flags:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                prev_path = os.environ["PATH"]
                os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if path_patched:
                os.environ["PATH"] = prev_path
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err
    kernel32.SetErrorMode(prev_error_mode)

from .version import version
from .convert import export
from .convert import dygraph2onnx
from .convert import load_parameter
from .convert import save_program

__version__ = version

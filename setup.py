#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# This file referred to github.com/onnx/onnx.git

from __future__ import annotations

import multiprocessing
import os
import platform
import shlex
import subprocess
import sys
from contextlib import contextmanager
from distutils import log, sysconfig
from shutil import which
from typing import ClassVar

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, "paddle2onnx")
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")

WINDOWS = os.name == "nt"

CMAKE = which("cmake3") or which("cmake")
MAKE = which("make")

################################################################################
# Global variables for controlling the build variant
################################################################################

# Default value is set to TRUE\1 to keep the settings same as the current ones.
# However going forward the recomemded way to is to set this to False\0
USE_MSVC_STATIC_RUNTIME = bool(os.getenv("USE_MSVC_STATIC_RUNTIME", "1") == "1")
ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE", "onnx")

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f"Can only cd to absolute path, got: {path}")
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.
    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ("jobs=", "j", "Specifies the number of jobs to use with make")
    ]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            # configure
            cmake_args = [
                CMAKE,
                f"-DPYTHON_INCLUDE_DIR={sysconfig.get_python_inc()}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DBUILD_PADDLE2ONNX_PYTHON=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DONNX_NAMESPACE={ONNX_NAMESPACE}",
                "-DPY_EXT_SUFFIX={}".format(
                    sysconfig.get_config_var("EXT_SUFFIX") or ""
                ),
                "-DPY_VERSION={}".format(
                    str(sys.version_info[0]) + "." + str(sys.version_info[1])
                ),
            ]
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
            cmake_args.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
            if WINDOWS:
                cmake_args.extend(
                    [
                        # we need to link with libpython on windows, so
                        # passing python version to window in order to
                        # find python in cmake
                        "-DPY_VERSION={}".format("{}.{}".format(*sys.version_info[:2])),
                    ]
                )
                if platform.architecture()[0] == "64bit":
                    cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:
                    cmake_args.extend(["-A", "Win32", "-T", "host=x86"])
            else:
                cmake_args.append(
                    f"-DPYTHON_LIBRARY={sysconfig.get_python_lib(standard_lib=True)}"
                )
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                log.info(f"Extra cmake args: {extra_cmake_args}")
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, "--build", os.curdir]
            if WINDOWS:
                build_args.extend(["--config", build_type])
                build_args.extend(["--", f"/maxcpucount:{self.jobs}"])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            subprocess.check_call(build_args)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()

    def build_extensions(self):
        build_lib = self.build_lib
        extension_dst_dir = os.path.join(build_lib, "paddle2onnx")
        os.makedirs(extension_dst_dir, exist_ok=True)

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if WINDOWS:
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir

            src = os.path.join(lib_path, filename)
            dst = os.path.join(
                os.path.realpath(self.build_lib), "paddle2onnx", filename
            )
            if platform.system() == "Darwin":
                command = f"install_name_tool -change @loader_path/../libs/ @loader_path/../paddle/base/libpaddle.so {src}"
                if os.system(command) != 0:
                    raise Exception(
                        f"Failed to change library paths using command: '{command}'"
                    )
            self.copy_file(src, dst)


cmdclass = {
    "cmake_build": cmake_build,
    "build_ext": build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(name="paddle2onnx.paddle2onnx_cpp2py_export", sources=[])
]

################################################################################
# Final
################################################################################

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

project(p2o_exec)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++11")

set(paddle2onnx_install_path "paddle2onnx-linux-x64-1.0.0rc")

find_library(paddle2onnx_lib
    NAMES paddle2onnx
    PATHS ${paddle2onnx_install_path}/lib/
    )
LINK_DIRECTORIES(${paddle2onnx_install_path})
add_executable(${PROJECT_NAME} cpp_export.cpp )
include_directories(${paddle2onnx_install_path}/include)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${paddle2onnx_lib})

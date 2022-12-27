project(p2o_exec)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++11")

set(paddle2onnx_install_path "paddle2onnx-linux-x64")
find_library(paddle2onnx_lib
    NAMES paddle2onnx
    PATHS ${paddle2onnx_install_path}/lib/
    )    
find_library(paddle2onnx_bundled_lib
    NAMES paddle2onnx_bundled
    PATHS ${paddle2onnx_install_path}/lib/
    )        
add_executable(${PROJECT_NAME} cpp_export.cpp )
include_directories(${paddle2onnx_install_path}/include)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${paddle2onnx_lib})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${paddle2onnx_bundled_lib})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS 
                      "-Wl,--whole-archive ${paddle2onnx_lib} -Wl,-no-whole-archive") 
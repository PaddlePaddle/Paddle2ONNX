cmake_minimum_required(VERSION 3.0)
project(PaddleX CXX C)

SET(TENSORRT_DIR "" CACHE PATH "Location of libraries")
SET(TENSORRT_HEADER "" CACHE PATH "Location of libraries")
SET(CUDA_DIR "" CACHE PATH "Location of libraries")
SET(GLOG_DIR "" CACHE PATH "Location of libraries")
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2  -fopenmp -std=c++11")

SET(PROJECT_ROOT_DIR  "../../" CACHE PATH  "root directory of project.")

if (NOT WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/paddlex_inference)
endif()

if (NOT WIN32)
    SET(YAML_BUILD_TYPE ON CACHE BOOL "yaml build shared library.")
else()
    SET(YAML_BUILD_TYPE OFF CACHE BOOL "yaml build shared library.")
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

include_directories("${PROJECT_ROOT_DIR}/")

include(${PROJECT_ROOT_DIR}/cmake/yaml-cpp.cmake)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

include_directories("${GFLAGS_DIR}/include")
link_directories("${GFLAGS_DIR}/lib")

include_directories("${GLOG_DIR}/include")
link_directories("${GLOG_DIR}/lib")

include_directories("${TENSORRT_HEADER}/include")
link_directories("${TENSORRT_DIR}/lib")

include_directories("${CUDA_DIR}/include")
link_directories("${CUDA_DIR}/lib")

find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)


set(DEPS ${TENSORRT_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} ${TENSORRT_DIR}/libnvonnxparser${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} ${TENSORRT_DIR}/libnvparsers${CMAKE_SHARED_LIBRARY_SUFFIX})

set(DEPS ${DEPS} ${CUDA_DIR}/lib/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
set(DEPS ${DEPS} ${OpenCV_LIBS})
set(DEPS ${DEPS} glog gflags yaml-cpp)

if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

add_library(tensorrt_inference STATIC  ${PROJECT_ROOT_DIR}/src/engine/tensorrt_engine.cpp ${PROJECT_ROOT_DIR}/src/postprocess/ppdet_post_proc.cpp ${PROJECT_ROOT_DIR}/src/preprocess/ppdet_pre_proc.cpp ${PROJECT_ROOT_DIR}/src/preprocess/preprocess.cpp  ${PROJECT_ROOT_DIR}/src/common/config.cpp  ${PROJECT_ROOT_DIR}/src/common/transforms.cpp ${PROJECT_ROOT_DIR}/src/postprocess/util/clipper.cpp ${PROJECT_ROOT_DIR}/src/postprocess/ppocr_post_proc.cpp ${PROJECT_ROOT_DIR}/src/preprocess/ppocr_pre_proc.cpp)

ADD_DEPENDENCIES(tensorrt_inference ext-yaml-cpp)
target_link_libraries(tensorrt_inference ${DEPS})

set(DEPS ${DEPS} tensorrt_inference)

add_executable(ppdet_tensorrt_infer ${PROJECT_ROOT_DIR}/demo/tensorrt_inference/ppdet_infer.cpp)
ADD_DEPENDENCIES(ppdet_tensorrt_infer ext-yaml-cpp)
target_link_libraries(ppdet_tensorrt_infer ${DEPS})


add_executable(ppocr_tensorrt_infer ${PROJECT_ROOT_DIR}/demo/tensorrt_inference/ppocr_infer.cpp)
ADD_DEPENDENCIES(ppocr_tensorrt_infer ext-yaml-cpp)
target_link_libraries(ppocr_tensorrt_infer ${DEPS})

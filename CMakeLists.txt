PROJECT(paddle2onnx C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

ADD_SUBDIRECTORY(paddle2onnx)

option(WITH_STATIC              "Compile Paddle2ONNX with  STATIC"               OFF)
option(PADDLE2ONNX_DEBUG "If open the debug log while converting model"          OFF)
option(MSVC_STATIC_CRT          "Compile Paddle2ONNX with  MSVC STATIC CRT"               ON)
set(ONNX_CUSTOM_PROTOC_PATH "" CACHE STRING "Set yourself Protobufc path")

if(PADDLE2ONNX_DEBUG)
  add_definitions(-DPADDLE2ONNX_DEBUG)
endif()
# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

# setting max opset version for onnx
# if you build from other version of onnx
# this should be modified
# refer to https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
add_definitions(-DMAX_ONNX_OPSET_VERSION=15)
add_definitions(-DPADDLE2ONNX_LIB)

# Third dependency: onnx
if(NOT TARGET onnx_proto)
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "paddle2onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

  if(ONNX_CUSTOM_PROTOC_PATH)
    if(WIN32)
      if(MSVC_STATIC_CRT)
        # MT
        set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
      else()
        # MD
        set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
      endif()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH};$ENV{PATH}")
    else()
      set(ONNX_CUSTOM_PROTOC_PATH "${ONNX_CUSTOM_PROTOC_PATH}:$ENV{PATH}")
    endif()
    set(ENV{PATH} ${ONNX_CUSTOM_PROTOC_PATH})
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third/onnx)
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third/optimizer)
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc ${PROJECT_SOURCE_DIR}/third/optimizer/onnxoptimizer/*.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/third/optimizer/onnxoptimizer/cpp2py_export.cc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
if (WITH_STATIC)
    ADD_LIBRARY(paddle2onnx STATIC ${ALL_SRCS})
    TARGET_LINK_LIBRARIES(paddle2onnx  p2o_paddle_proto onnx )
else ()
    ADD_LIBRARY(paddle2onnx SHARED ${ALL_SRCS})
    TARGET_LINK_LIBRARIES(paddle2onnx p2o_paddle_proto onnx)
endif()

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/paddle2onnx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "converter.h"
)
install(
    TARGETS paddle2onnx
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (BUILD_PADDLE2ONNX_EXE)
  remove_definitions(-DPADDLE2ONNX_LIB)
  ADD_EXECUTABLE(p2o_exec p2o_exec.cpp)
  if (WITH_STATIC)
    TARGET_LINK_LIBRARIES(p2o_exec -Wl,--whole-archive paddle2onnx -Wl,--no-whole-archive)
  else()
    TARGET_LINK_LIBRARIES(p2o_exec paddle2onnx)
  endif()
endif (BUILD_PADDLE2ONNX_EXE)

if(BUILD_PADDLE2ONNX_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  add_library(paddle2onnx_cpp2py_export MODULE ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc ${ALL_SRCS})
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(paddle2onnx_cpp2py_export PRIVATE
                             $<BUILD_INTERFACE:${ONNX_ROOT}>
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  # pybind11 is a header only lib
  find_package(pybind11 2.2)
  if(pybind11_FOUND)
    target_include_directories(paddle2onnx_cpp2py_export PUBLIC
      ${pybind11_INCLUDE_DIRS})
  else()
    if(EXISTS ${PROJECT_SOURCE_DIR}/third/pybind11/include/pybind11/pybind11.h)
      target_include_directories(paddle2onnx_cpp2py_export PUBLIC
      ${PROJECT_SOURCE_DIR}/third/pybind11/include)
    else()
      message(FATAL_ERROR "cannot find pybind")
    endif()
  endif()

  if(APPLE)
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    # whole-archive linker option not available on AIX
    target_sources(paddle2onnx_cpp2py_export
                          PRIVATE $<TARGET_OBJECTS:onnx>)
  else()
    # Assume everything else is like gcc
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx>
                                  "-Wl,--no-whole-archive")
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()

  target_link_libraries(paddle2onnx_cpp2py_export PRIVATE p2o_paddle_proto onnx)

  if(MSVC)
    target_link_libraries(paddle2onnx_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(paddle2onnx_cpp2py_export
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    if(ONNX_USE_PROTOBUF_SHARED_LIBS)
      target_compile_options(onnx_cpp2py_export
                             PRIVATE /wd4251 # 'identifier' : class 'type1' needs to
                                             # have dll-interface to be used by
                                             # clients of class 'type2'
                            )
    endif()
    add_msvc_runtime_flag(paddle2onnx_cpp2py_export)
    add_onnx_global_defines(paddle2onnx_cpp2py_export)
  endif()
endif()



#############################CMAKE FOR DEPLOYKIT################################
option(ENABLE_PADDLE_FRONTEND "if to enable PaddlePaddle frontend to support load paddle model in deploykit." ON)
option(ENABLE_ORT_BACKEND "if to enable onnxruntime backend." OFF)
option(ORT_DIRECTORY "if build onnxruntime backend, need to define path of onnxruntime library.")

option(ENABLE_TRT_BACKEND "if to enable tensorrt backend." OFF)
option(CUDA_DIRECTORY "if build tensorrt backend, need to define path of cuda library.")
option(TRT_DIRECTORY "if build tensorrt backend, need to define path of tensorrt library.")

option(BUILD_DEMO "if to build the deploykit demo code." ON)
option(BUILD_DEPLOYKIT_PYTHON "if build python lib for deploykit." OFF)

if(BUILD_DEPLOYKIT)
file(GLOB_RECURSE ALL_KIT_SRCS ${PROJECT_SOURCE_DIR}/deploykit/*.cc)
file(GLOB_RECURSE KIT_ORT_SRCS ${PROJECT_SOURCE_DIR}/deploykit/backends/ort/*.cc)
file(GLOB_RECURSE KIT_TRT_SRCS ${PROJECT_SOURCE_DIR}/deploykit/backends/tensorrt/*.cc ${PROJECT_SOURCE_DIR}/deploykit/backends/tensorrt/*.cpp)
list(REMOVE_ITEM ALL_KIT_SRCS ${KIT_ORT_SRCS} ${KIT_TRT_SRCS})
list(REMOVE_ITEM ALL_KIT_SRCS ${PROJECT_SOURCE_DIR}/deploykit/deploy_py_export.cc)

set(BACKEND_LIBS "")
if(ENABLE_ORT_BACKEND)
  add_definitions(-DENABLE_ORT_BACKEND)
  list(APPEND ALL_KIT_SRCS ${KIT_ORT_SRCS})
  include_directories(${ORT_DIRECTORY}/include)
  find_library(ONNXRUNTIME_LIB onnxruntime ${ORT_DIRECTORY}/lib)
  list(APPEND BACKEND_LIBS ${ONNXRUNTIME_LIB})
endif()

if(ENABLE_TRT_BACKEND)
  add_definitions(-DENABLE_TRT_BACKEND)
  list(APPEND ALL_KIT_SRCS ${KIT_TRT_SRCS})
  include_directories(${CUDA_DIRECTORY}/include)
  include_directories(${TRT_DIRECTORY}/include)
  include_directories(${PROJECT_SOURCE_DIR}/deploykit/backends/tensorrt/common)
  find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib64)
  find_library(TRT_INFER_LIB nvinfer ${TRT_DIRECTORY}/lib)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_DIRECTORY}/lib)
  find_library(TRT_CAFFE_LIB nvcaffe_parser ${TRT_DIRECTORY}/lib)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_DIRECTORY}/lib)
  list(APPEND BACKEND_LIBS ${CUDA_LIB} ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_CAFFE_LIB} ${TRT_PLUGIN_LIB})
endif()

ADD_LIBRARY(deploykit SHARED ${ALL_KIT_SRCS})
target_link_libraries(deploykit PUBLIC ${BACKEND_LIBS})

if(ENABLE_PADDLE_FRONTEND)
  add_definitions(-DENABLE_PADDLE_FRONTEND)
  target_link_libraries(deploykit PUBLIC paddle2onnx)
endif(ENABLE_PADDLE_FRONTEND)

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/deploykit
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "converter.h"
)
install(
  TARGETS deploykit
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
  FILES ${BACKEND_LIBS}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(BUILD_DEMO)
  add_executable(resnet50_exe ${PROJECT_SOURCE_DIR}/deploykit_demo/cpp/resnet50.cc)
  target_link_libraries(resnet50_exe PUBLIC deploykit)
endif()

if(BUILD_DEPLOYKIT_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  add_library(deploykit_cpp2py_export MODULE ${PROJECT_SOURCE_DIR}/deploykit/deploy_py_export.cc)
  set_target_properties(deploykit_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(deploykit_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(deploykit_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(deploykit_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(deploykit_cpp2py_export PRIVATE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  # pybind11 is a header only lib
  find_package(pybind11 2.2)
  if(pybind11_FOUND)
    target_include_directories(deploykit_cpp2py_export PUBLIC
      ${pybind11_INCLUDE_DIRS})
  else()
    if(EXISTS ${PROJECT_SOURCE_DIR}/third/pybind11/include/pybind11/pybind11.h)
      target_include_directories(deploykit_cpp2py_export PUBLIC
      ${PROJECT_SOURCE_DIR}/third/pybind11/include)
    else()
      message(FATAL_ERROR "cannot find pybind")
    endif()
  endif()

  if(APPLE)
    set_target_properties(deploykit_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  if(APPLE)
    target_link_libraries(deploykit_cpp2py_export PUBLIC deploykit)
  elseif(WIN32)
    target_link_libraries(deploykit_cpp2py_export PUBLIC deploykit)
  else()
    target_link_libraries(deploykit_cpp2py_export PUBLIC deploykit)
  endif()

  if(MSVC)
    target_link_libraries(deploykit_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(deploykit_cpp2py_export
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    target_compile_options(deploykit_cpp2py_export PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT> $<$<CONFIG:Debug>:/MTd>)
  endif()
endif(BUILD_DEPLOYKIT_PYTHON)

endif(BUILD_DEPLOYKIT)
################################################################################

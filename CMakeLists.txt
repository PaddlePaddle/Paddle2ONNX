cmake_minimum_required(VERSION 3.16)
project(paddle2onnx C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
link_libraries(
  "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>"
)

option(WITH_STATIC "Compile Paddle2ONNX with STATIC" OFF)
option(PADDLE2ONNX_DEBUG "If open the debug log while converting model" OFF)
option(MSVC_STATIC_CRT "Compile Paddle2ONNX with MSVC STATIC CRT" OFF)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  add_definitions(-DPADDLE_WITH_TESTING)
endif()

if(PADDLE2ONNX_DEBUG)
  add_definitions(-DPADDLE2ONNX_DEBUG)
endif()

add_definitions(-DMAX_ONNX_OPSET_VERSION=25)
add_definitions(-DPADDLE2ONNX_LIB)

set(WITH_PADDLE2ONNX_STATIC_INTERNAL OFF)
if(WITH_STATIC)
  set(WITH_PADDLE2ONNX_STATIC_INTERNAL
      ON
      CACHE BOOL "" FORCE)
  add_definitions(-DWITH_PADDLE2ONNX_STATIC_INTERNAL_AT_COMPILING)
endif()

include(utils)
configure_file(${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h.in
               ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h)

# --- Dependencies ---

# Protobuf (try CONFIG mode first, fall back to module mode)
find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()

include(FetchContent)

# ONNX (built from source for protobuf compatibility)
if(NOT ONNX_NAMESPACE)
  set(ONNX_NAMESPACE "onnx")
endif()
add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)
set(ONNX_ML ON CACHE BOOL "" FORCE)
set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG v1.20.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(onnx)

# onnx-optimizer: fetch source only, compile directly
FetchContent_Declare(
  onnx_optimizer
  GIT_REPOSITORY https://github.com/onnx/optimizer.git
  GIT_TAG b3a4611861734e0731bbcc2bed1f080139e4988b # v0.3.19
  GIT_SHALLOW FALSE
  PATCH_COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_optimizer.cmake
)
FetchContent_Populate(onnx_optimizer)

# pybind11 (system-installed)
find_package(pybind11 REQUIRED)

# glog (system-installed)
if(MSVC)
  set(WITH_GFLAGS
      OFF
      CACHE BOOL "Disable gflags support in glog")
endif()
find_package(glog REQUIRED)

# generate Paddle2ONNX proto files
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${onnx_optimizer_SOURCE_DIR})

# Collect sources: paddle2onnx + onnx-optimizer (compiled in)
file(GLOB_RECURSE OPTIMIZER_SRCS
     ${onnx_optimizer_SOURCE_DIR}/onnxoptimizer/*.cc)
file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc)
list(APPEND ALL_SRCS ${OPTIMIZER_SRCS})
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)
list(REMOVE_ITEM ALL_SRCS
     ${onnx_optimizer_SOURCE_DIR}/onnxoptimizer/cpp2py_export.cc)

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" PADDLE2ONNX_VERSION)
string(STRIP "${PADDLE2ONNX_VERSION}" PADDLE2ONNX_VERSION)

# find Python site-packages and PaddlePaddle
set(PYTHON_SITE_PACKAGES "")
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import site; print(';'.join(site.getsitepackages()))"
  OUTPUT_VARIABLE SITE_PACKAGES_LIST
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE ";" " " SITE_PACKAGES_LIST "${SITE_PACKAGES_LIST}")
separate_arguments(SITE_PACKAGES_LIST)
foreach(SITE_PACKAGES IN LISTS SITE_PACKAGES_LIST)
  if(EXISTS "${SITE_PACKAGES}" AND SITE_PACKAGES MATCHES "site-packages")
    string(REPLACE "\\" "/" PYTHON_SITE_PACKAGES "${SITE_PACKAGES}")
    message(STATUS "Found valid site-packages path: ${PYTHON_SITE_PACKAGES}.")
    break()
  endif()
endforeach()
if(NOT PYTHON_SITE_PACKAGES)
  message(FATAL_ERROR "No valid site-packages path found.")
endif()

if(MSVC)
  set(PADDLE_LIB "${PYTHON_SITE_PACKAGES}/paddle/base/libpaddle.lib")
  set(COMMON_LIB "${PYTHON_SITE_PACKAGES}/paddle/libs/common.lib")
else()
  set(PADDLE_LIB "${PYTHON_SITE_PACKAGES}/paddle/base/libpaddle.so")
endif()
if(EXISTS ${PADDLE_LIB})
  message(STATUS "Found libpaddle: ${PADDLE_LIB}.")
else()
  message(
    FATAL_ERROR
      "Can not find libpaddle at ${PYTHON_SITE_PACKAGES}/paddle/base!")
endif()
set(PADDLE_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/paddle/include/)
include_directories(${PADDLE_INCLUDE_DIR})

if(MSVC)
  add_custom_target(
    pir_patch ALL
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_SOURCE_DIR}/patch/paddle/interface_support.h
      ${PADDLE_INCLUDE_DIR}/paddle/pir/include/core/interface_support.h
    COMMENT "PIR patch for MSVC")
endif()

if(WITH_STATIC)
  add_library(paddle2onnx_dummy STATIC ${ALL_SRCS})
  if(NOT MSVC)
    set_target_properties(paddle2onnx_dummy PROPERTIES COMPILE_FLAGS
                                                       "-fvisibility=hidden")
  endif()
  target_link_libraries(paddle2onnx_dummy p2o_paddle_proto onnx)
  bundle_static_library(paddle2onnx_dummy paddle2onnx_static bundle_paddle2onnx)
else()
  add_library(paddle2onnx SHARED ${ALL_SRCS})
  if(APPLE)
    set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS
                                                 "-undefined dynamic_lookup")
  elseif(MSVC)
    message("------ BUILD WITH MSVC --------")
  else()
    set_target_properties(paddle2onnx PROPERTIES COMPILE_FLAGS
                                                 "-fvisibility=hidden")
    set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS
                                                 "-Wl,--exclude-libs,ALL")
    set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS_RELEASE -s)
  endif()
  set_target_properties(paddle2onnx PROPERTIES VERSION ${PADDLE2ONNX_VERSION})
  target_link_libraries(
    paddle2onnx
    ${PADDLE_LIB}
    ${COMMON_LIB}
    p2o_paddle_proto
    onnx
    pybind11::embed
    glog::glog)
endif()

if(WIN32)
  install(
    TARGETS paddle2onnx
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib)
else()
  if(WITH_STATIC)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpaddle2onnx_static.a
            DESTINATION lib)
  else()
    install(TARGETS paddle2onnx LIBRARY DESTINATION lib)
  endif()
endif()
install(FILES ${PROJECT_SOURCE_DIR}/paddle2onnx/converter.h
              ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h
        DESTINATION include/paddle2onnx)

if(BUILD_PADDLE2ONNX_PYTHON)
  if(APPLE)
    find_program(INSTALL_NAME_TOOL_EXECUTABLE install_name_tool)
    if(NOT INSTALL_NAME_TOOL_EXECUTABLE)
      message(FATAL_ERROR "install_name_tool not found, please check.\n")
    endif()
  endif()
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  add_library(paddle2onnx_cpp2py_export MODULE
              ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc ${ALL_SRCS})
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(paddle2onnx_cpp2py_export PROPERTIES SUFFIX
                                                             ${PY_EXT_SUFFIX})
  set_target_properties(paddle2onnx_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(
    paddle2onnx_cpp2py_export
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:include>
            ${PYTHON_INCLUDE_DIR}
            ${PADDLE_INCLUDE_DIR}
            ${pybind11_INCLUDE_DIRS})

  if(APPLE)
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
  elseif(MSVC)
    target_link_libraries(paddle2onnx_cpp2py_export
                          PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
  else()
    target_link_libraries(
      paddle2onnx_cpp2py_export
      PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx>
              "-Wl,--no-whole-archive")
    set_target_properties(paddle2onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  endif()

  target_link_libraries(
    paddle2onnx_cpp2py_export PRIVATE ${PADDLE_LIB} ${COMMON_LIB}
                                      p2o_paddle_proto onnx glog::glog)

  if(MSVC)
    target_link_libraries(paddle2onnx_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(
      paddle2onnx_cpp2py_export
      PRIVATE /MP /wd4244 /wd4267 /wd4996 ${EXTRA_FLAGS})
  endif()
endif()

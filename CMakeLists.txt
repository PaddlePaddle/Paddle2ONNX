cmake_minimum_required(VERSION 3.16)
project(paddle2onnx C CXX)
# ONNX 1.20 requires C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Build the libraries with - fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Always link with libstdc++ fs.a when using GCC 8.
link_libraries(
  "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>"
)

option(WITH_STATIC "Compile Paddle2ONNX with  STATIC" OFF)
option(PADDLE2ONNX_DEBUG "If open the debug log while converting model" OFF)
option(MSVC_STATIC_CRT "Compile Paddle2ONNX with  MSVC STATIC CRT" OFF)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  add_definitions(-DPADDLE_WITH_TESTING)
endif()

if(PADDLE2ONNX_DEBUG)
  add_definitions(-DPADDLE2ONNX_DEBUG)
endif()

# Set max opset version for onnx if you build from other version of onnx this
# should be modified.
add_definitions(-DMAX_ONNX_OPSET_VERSION=25)
add_definitions(-DPADDLE2ONNX_LIB)

# Internal flags for convert.h.in
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

# Third dependency : onnx
if(NOT TARGET onnx_proto)
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
endif()

# generate Paddle2ONNX proto files
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/optimizer)

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc
     ${PROJECT_SOURCE_DIR}/third_party/optimizer/onnx/optimizer/*.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" PADDLE2ONNX_VERSION)
string(STRIP "${PADDLE2ONNX_VERSION}" PADDLE2ONNX_VERSION)

if(WITH_STATIC)
  # Here, we use a dummy target (paddle2onnx_dummy) to form a build dependency
  # tree for paddle2onnx_static lib.
  add_library(paddle2onnx_dummy STATIC ${ALL_SRCS})

  if(APPLE)
    set_target_properties(paddle2onnx_dummy PROPERTIES COMPILE_FLAGS
                                                       "-fvisibility=hidden")
  elseif(MSVC)
    message("------ BUILD WITH MSVC --------")
  else()
    set_target_properties(paddle2onnx_dummy PROPERTIES COMPILE_FLAGS
                                                       "-fvisibility=hidden")
  endif()
  target_link_libraries(paddle2onnx_dummy p2o_paddle_proto onnx)
  # Bundle paddle2onnx static lib here
  bundle_static_library(paddle2onnx_dummy paddle2onnx_static bundle_paddle2onnx)
else()
  add_library(paddle2onnx SHARED ${ALL_SRCS})

  # add pybind11
  add_subdirectory(third_party/pybind11)

  # add glog
  if(MSVC)
    set(WITH_GFLAGS
        OFF
        CACHE BOOL "Disable gflags support in glog")
  endif()
  add_subdirectory(third_party/glog)

  # find Python site-packages dir
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
  # add paddle include dir
  set(PADDLE_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/paddle/include/)
  include_directories(${PADDLE_INCLUDE_DIR})
  if(MSVC)
    # PIR patch
    add_custom_target(
      pir_patch ALL
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/patch/paddle/interface_support.h
        ${PADDLE_INCLUDE_DIR}/paddle/pir/include/core/interface_support.h
      COMMENT
        "Replace file ${PADDLE_INCLUDE_DIR}/paddle/pir/include/core/interface_support.h with ${CMAKE_SOURCE_DIR}/patch/paddle/interface_support.h"
    )
  endif()
  if(APPLE)
    set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS
                                                 "-undefined dynamic_lookup")
  elseif(MSVC)
    message("------ BUILD WITH MSVC --------")
    # remove `interface` macro defined in MSVC
    # target_compile_options(paddle2onnx PRIVATE /Uinterface)
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
    PRIVATE $<BUILD_INTERFACE:${ONNX_ROOT}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:include>
            ${PYTHON_INCLUDE_DIR}
            ${PADDLE_INCLUDE_DIR})

  if(EXISTS
     ${PROJECT_SOURCE_DIR}/third_party/pybind11/include/pybind11/pybind11.h)
    target_include_directories(
      paddle2onnx_cpp2py_export
      PUBLIC ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
  else()
    message(FATAL_ERROR "cannot find pybind")
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
  else()
    # Assume everything else is like gcc
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
      PRIVATE /MP
              /wd4244 # 'argument': conversion from 'google::protobuf::uint64'
                      # to 'int', possible loss of data
              /wd4267 # Conversion from 'size_t' to 'int', possible loss of data
              /wd4996 # The second parameter is ignored.
              ${EXTRA_FLAGS})
    if(ONNX_USE_PROTOBUF_SHARED_LIBS)
      target_compile_options(
        onnx_cpp2py_export
        PRIVATE /wd4251 # 'identifier' : class 'type1' needs to have
                        # dll-interface to be used by clients of class 'type2'
      )
    endif()
    add_msvc_runtime_flag(paddle2onnx_cpp2py_export)
    add_onnx_global_defines(paddle2onnx_cpp2py_export)
  endif()
endif()

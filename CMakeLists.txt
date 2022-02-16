PROJECT(paddle2onnx C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

ADD_SUBDIRECTORY(paddle2onnx)

#set(ONNX_USE_LITE_PROTO ON)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# setting max opset version for onnx
# if you build from other version of onnx
# this should be modified
# refer to https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
add_definitions(-DMAX_ONNX_OPSET_VERSION=15)

# Third dependency: onnx
if(NOT TARGET onnx_proto)
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "paddle2onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third/onnx)
endif()

include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)

ADD_LIBRARY(paddle2onnx SHARED ${ALL_SRCS})
TARGET_INCLUDE_DIRECTORIES(paddle2onnx
   PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
   PUBLIC ${PROTOBUF_INCLUDE_DIRS}
)
TARGET_LINK_LIBRARIES(paddle2onnx p2o_paddle_proto onnx)

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
#install(
#    TARGETS paddle2onnx
#)

if (BUILD_PADDLE2ONNX_EXE)
  ADD_EXECUTABLE(p2o_exec p2o_exec.cpp)
  TARGET_INCLUDE_DIRECTORIES(p2o_exec
     PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
     PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
     PUBLIC ${PROTOBUF_INCLUDE_DIRS}
  )
  TARGET_LINK_LIBRARIES(p2o_exec paddle2onnx)
endif (BUILD_PADDLE2ONNX_EXE)

if(BUILD_PADDLE2ONNX_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
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

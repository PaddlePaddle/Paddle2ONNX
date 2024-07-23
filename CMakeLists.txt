CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(paddle2onnx C CXX)
# ONNX 1.16 requires C++17
set(CMAKE_CXX_STANDARD 17)
# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Always link with libstdc++fs.a when using GCC 8. See https://discourse.cmake.org/t/correct-way-to-link-std-filesystem-with-gcc-8/4121/5 for detail.
link_libraries( "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>" )

option(WITH_STATIC "Compile Paddle2ONNX with  STATIC" OFF)
option(PADDLE2ONNX_DEBUG "If open the debug log while converting model" OFF)
option(MSVC_STATIC_CRT "Compile Paddle2ONNX with  MSVC STATIC CRT" ON)

if (PADDLE2ONNX_DEBUG)
    add_definitions(-DPADDLE2ONNX_DEBUG)
endif ()

# Set max opset version for onnx if you build from other version of onnx this should be modified.
# Refer to https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
add_definitions(-DMAX_ONNX_OPSET_VERSION=19)
add_definitions(-DPADDLE2ONNX_LIB)

# Internal flags for convert.h.in
set(WITH_PADDLE2ONNX_STATIC_INTERNAL OFF)
if (WITH_STATIC)
    set(WITH_PADDLE2ONNX_STATIC_INTERNAL ON CACHE BOOL "" FORCE)
    add_definitions(-DWITH_PADDLE2ONNX_STATIC_INTERNAL_AT_COMPILING)
endif ()

include(cmake/utils.cmake)
configure_file(${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h.in ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h)

# Third dependency: onnx
if (NOT TARGET onnx_proto)
    if (NOT ONNX_NAMESPACE)
        set(ONNX_NAMESPACE "onnx")
    endif ()
    add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
endif ()

# generate Paddle2ONNX proto files
add_subdirectory(${PROJECT_SOURCE_DIR}/paddle2onnx/proto)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/optimizer)

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/*.cc ${PROJECT_SOURCE_DIR}/third_party/optimizer/onnxoptimizer/*.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc)
list(REMOVE_ITEM ALL_SRCS ${PROJECT_SOURCE_DIR}/third_party/optimizer/onnxoptimizer/cpp2py_export.cc)

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" PADDLE2ONNX_VERSION)
string(STRIP "${PADDLE2ONNX_VERSION}" PADDLE2ONNX_VERSION)

if (WITH_STATIC)
    # Here, we use a dummy target (paddle2onnx_dummy)
    # to form a build dependency tree for paddle2onnx_static lib.
    ADD_LIBRARY(paddle2onnx_dummy STATIC ${ALL_SRCS})

    if (APPLE)
        set_target_properties(paddle2onnx_dummy PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    elseif (MSVC)
        message("------ BUILD WITH MSVC --------")
    else ()
        set_target_properties(paddle2onnx_dummy PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    endif ()
    target_link_libraries(paddle2onnx_dummy p2o_paddle_proto onnx)
    # Bundle paddle2onnx static lib here
    bundle_static_library(paddle2onnx_dummy paddle2onnx_static bundle_paddle2onnx)
else ()
    ADD_LIBRARY(paddle2onnx SHARED ${ALL_SRCS})
    if (APPLE)
        set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    elseif (MSVC)
        message("------ BUILD WITH MSVC --------")
    else ()
        set_target_properties(paddle2onnx PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
        set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
        set_target_properties(paddle2onnx PROPERTIES LINK_FLAGS_RELEASE -s)
    endif ()
    set_target_properties(paddle2onnx PROPERTIES VERSION ${PADDLE2ONNX_VERSION})
    target_link_libraries(paddle2onnx p2o_paddle_proto onnx)
endif ()

if (WIN32)
    install(
            TARGETS paddle2onnx
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )
else ()
    if (WITH_STATIC)
        install(
                FILES ${CMAKE_CURRENT_BINARY_DIR}/libpaddle2onnx_static.a
                DESTINATION lib
        )
    else ()
        install(
                TARGETS paddle2onnx
                LIBRARY DESTINATION lib
        )
    endif ()
endif ()
install(
        FILES
        ${PROJECT_SOURCE_DIR}/paddle2onnx/converter.h
        ${PROJECT_SOURCE_DIR}/paddle2onnx/mappers_registry.h
        DESTINATION include/paddle2onnx
)

if (BUILD_PADDLE2ONNX_PYTHON)
    if ("${PY_EXT_SUFFIX}" STREQUAL "")
        if (MSVC)
            set(PY_EXT_SUFFIX ".pyd")
        else ()
            set(PY_EXT_SUFFIX ".so")
        endif ()
    endif ()

    add_library(paddle2onnx_cpp2py_export MODULE ${PROJECT_SOURCE_DIR}/paddle2onnx/cpp2py_export.cc ${ALL_SRCS})
    set_target_properties(paddle2onnx_cpp2py_export PROPERTIES PREFIX "")
    set_target_properties(paddle2onnx_cpp2py_export PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    set_target_properties(paddle2onnx_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(paddle2onnx_cpp2py_export PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_include_directories(paddle2onnx_cpp2py_export PRIVATE
            $<BUILD_INTERFACE:${ONNX_ROOT}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:include>
            ${PYTHON_INCLUDE_DIR})

    if (EXISTS ${PROJECT_SOURCE_DIR}/third_party/pybind11/include/pybind11/pybind11.h)
        target_include_directories(paddle2onnx_cpp2py_export PUBLIC ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
    else ()
        message(FATAL_ERROR "cannot find pybind")
    endif ()

    if (APPLE)
        set_target_properties(paddle2onnx_cpp2py_export PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
        target_link_libraries(paddle2onnx_cpp2py_export PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
    elseif (MSVC)
        # In MSVC, we will add whole archive in default
        target_link_libraries(paddle2onnx_cpp2py_export PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
    else ()
        # Assume everything else is like gcc
        target_link_libraries(paddle2onnx_cpp2py_export PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx> "-Wl,--no-whole-archive")
        set_target_properties(paddle2onnx_cpp2py_export PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
    endif ()

    target_link_libraries(paddle2onnx_cpp2py_export PRIVATE p2o_paddle_proto onnx)

    if (MSVC)
        target_link_libraries(paddle2onnx_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
        target_compile_options(paddle2onnx_cpp2py_export
                PRIVATE /MP
                /wd4244 # 'argument': conversion from 'google::protobuf::uint64' to 'int', possible loss of data
                /wd4267 # Conversion from 'size_t' to 'int', possible loss of data
                /wd4996 # The second parameter is ignored.
                ${EXTRA_FLAGS})
        if (ONNX_USE_PROTOBUF_SHARED_LIBS)
            target_compile_options(onnx_cpp2py_export
                    PRIVATE /wd4251 # 'identifier' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
            )
        endif ()
        add_msvc_runtime_flag(paddle2onnx_cpp2py_export)
        add_onnx_global_defines(paddle2onnx_cpp2py_export)
    endif ()
endif ()

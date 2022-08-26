cmake_minimum_required(VERSION 3.4.1)

project(Bisenet_demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ~/opt/rknpu2-master/runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ~/opt/rknpu2-master/runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  if (CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ~/opt/rknpu2-master/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ~/opt/rknpu2-master/examples/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    set(OpenCV_DIR ~/opt/rknpu2-master/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  endif()
endif()
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(Bisenet_demo
        src/main.cpp
        ../../lib/image_tools.cpp
        src/rknn_config.cpp
        src/utility.cpp
        )
include_directories(
        ../../lib
        ./src
)
target_link_libraries(Bisenet_demo
        ${RKNN_RT_LIB}
        ${OpenCV_LIBS}
        )


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/Bisenet_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS Bisenet_demo DESTINATION ./)
install(DIRECTORY weights DESTINATION ./)
install(DIRECTORY images DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)

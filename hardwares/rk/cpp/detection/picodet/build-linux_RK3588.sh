set -e

TARGET_SOC="rk3588"
GCC_COMPILER=~/opt/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu/bin/aarch64-rockchip-linux-gnu
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# create build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64
mkdir -p ${BUILD_DIR}

# create install
mkdir -p ./install

# build and make
cd ${BUILD_DIR}
cmake ../.. \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++
make -j4
make install
cd -
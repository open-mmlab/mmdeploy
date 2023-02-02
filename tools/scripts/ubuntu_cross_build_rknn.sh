#!/bin/bash
# set -ex
# get appropriate proc number: max(1, nproc-3)
good_nproc() {
  num=`nproc`
  num=`expr $num - 3`
  if [ $num -lt 1 ];then
    return 1
  fi
  return ${num}
}

install_rknpu_toolchain() {
  # install gcc cross compiler
  ubuntu_version=`cat /etc/issue`
  ubuntu_major_version=`echo "$ubuntu_version" | grep -oP '\d{2}' | head -n 1`

  if [ "$ubuntu_major_version" -lt 18 ]; then
    echo "ubuntu 18.04 is minimum requirement, but got $ubuntu_version"
    wget wget https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
    tar -xvf gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz
    sudo ln -sf $(pwd)/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc /usr/bin/arm-linux-gnueabihf-gcc
    sudo ln -sf $(pwd)/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ /usr/bin/arm-linux-gnueabihf-g++
  else
    sudo apt install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
  fi
  arm-linux-gnueabihf-gcc --version
  arm-linux-gnueabihf-g++ --version

  # install rknpu
  git clone https://github.com/rockchip-linux/rknpu
  export RKNPU_DIR=$(pwd)/rknpu

  sudo apt install wget git git-lfs

  python3 -m pip install cmake==3.22.0

  echo 'export PATH=~/.local/bin:${PATH}' >> ~/mmdeploy.env
  export PATH=~/.local/bin:${PATH}
}

install_rknpu2_toolchain() {
  sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
  aarch64-linux-gnu-g++ --version
  aarch64-linux-gnu-gcc --version
  git clone https://github.com/Caesar-github/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu.git
  git clone https://github.com/rockchip-linux/rknpu2.git
  export RKNN_TOOL_CHAIN=$(pwd)/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu
  export LD_LIBRARY_PATH=$RKNN_TOOL_CHAIN/lib64:$LD_LIBRARY_PATH
  export RKNPU2_DIR=$(pwd)/rknpu2
}

build_ocv_arm_gnueabi() {
  if [ ! -e "opencv" ];then
    git clone https://github.com/opencv/opencv --depth=1 --branch=4.6.0 --recursive
  fi
  if [ ! -e "opencv/build_arm_gnueabi" ];then
    mkdir -p opencv/build_arm_gnueabi
  fi
  cd opencv/build_arm_gnueabi
  rm -rf CMakeCache.txt
  cmake .. -DCMAKE_INSTALL_PREFIX=install -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/arm-gnueabi.toolchain.cmake \
    -DBUILD_TIFF=ON -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  good_nproc
  jobs=$?
  make -j${jobs} && make install
  export OPENCV_PACKAGE_DIR=$(pwd)/install/lib/cmake/opencv4
  cd -
}

build_ocv_aarch64() {
  if [ ! -e "opencv" ];then
    git clone https://github.com/opencv/opencv --depth=1 --branch=4.6.0 --recursive
  fi
  if [ ! -e "opencv/build_aarch64" ];then
    mkdir -p opencv/build_aarch64
  fi
  cd opencv/build_aarch64
  rm -rf CMakeCache.txt
  cmake .. -DCMAKE_INSTALL_PREFIX=install -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake \
    -DBUILD_PERF_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  good_nproc
  jobs=$?
  make -j${jobs} && make install
  export OPENCV_PACKAGE_DIR=$(pwd)/install/lib/cmake/opencv4
  cd -
}

build_mmdeploy_with_rknpu() {
  git submodule init
  git submodule update

  if [ ! -e "build_rknpu" ];then
    mkdir build_rknpu
  fi
  cd build_rknpu

  rm -rf CMakeCache.txt
  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/arm-linux-gnueabihf.cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_BACKENDS="rknn" \
    -DRKNPU_DEVICE_DIR="${RKNPU_DIR}"/rknn/rknn_api/librknn_api \
    -DOpenCV_DIR="${OPENCV_PACKAGE_DIR}"

  good_nproc
  jobs=$?
  make -j${jobs} && make install

  ls -lah install/bin/*
}

build_mmdeploy_with_rknpu2() {
  git submodule init
  git submodule update
  device_model=$1
  if [ ! -e "build_rknpu2" ];then
      mkdir build_rknpu2
  fi
  cd build_rknpu2

  rm -rf CMakeCache.txt
  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/rknpu2-linux-gnu.cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_BACKENDS="rknn" \
    -DRKNPU2_DEVICE_DIR="${RKNPU2_DIR}/runtime/${device_model}" \
    -DOpenCV_DIR="${OPENCV_PACKAGE_DIR}"

  good_nproc
  jobs=$?
  make -j${jobs} && make install

  ls -lah install/bin/*
}

print_success() {
  echo "----------------------------------------------------------------------"
  echo "Cross build finished, PLS copy bin/model/test_data to the device.. QVQ"
  echo "----------------------------------------------------------------------"
}

echo "the current workspace: $(pwd)"

if [ ! -e "../mmdeploy-dep" ];then
  mkdir ../mmdeploy-dep
fi
cd ../mmdeploy-dep

device_model=$(echo "$1" | tr [:lower:] [:upper:])
case "$device_model" in
  RK1808|RK1806|RV1109|RV1126)
    install_rknpu_toolchain
    build_ocv_arm_gnueabi
    cd ../mmdeploy
    build_mmdeploy_with_rknpu
    ;;
  RK3566|RK3568)
    install_rknpu2_toolchain
    build_ocv_aarch64
    cd ../mmdeploy
    build_mmdeploy_with_rknpu2 "RK356X"
    ;;
  RK3588|RV1106)
    install_rknpu2_toolchain
    build_ocv_aarch64
    cd ../mmdeploy
    build_mmdeploy_with_rknpu2 "$device_model"
    ;;
  *)
    echo "mmdeploy doesn't support rockchip '$1' yet"
    exit 1
    ;;
esac

print_success

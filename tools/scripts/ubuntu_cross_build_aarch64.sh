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

install_tools() {
  sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
  aarch64-linux-gnu-g++ --version
  aarch64-linux-gnu-gcc --version
  aarch64-linux-gnu-ld --version

  sudo apt install wget git git-lfs

  python3 -m pip install cmake==3.22.0

  echo 'export PATH=~/.local/bin:${PATH}' >> ~/mmdeploy.env
  export PATH=~/.local/bin:${PATH}
}

build_ocv() {
  if [ ! -e "opencv" ];then
    git clone https://github.com/opencv/opencv --depth=1 --branch=4.6.0 --recursive
  fi
  if [ ! -e "opencv/platforms/linux/cross_build_aarch64" ];then
    mkdir opencv/platforms/linux/cross_build_aarch64
  fi
  cd opencv/platforms/linux/cross_build_aarch64
  rm -rf CMakeCache.txt
  cmake ../../.. -DCMAKE_INSTALL_PREFIX=/tmp/ocv-aarch64 -DCMAKE_TOOLCHAIN_FILE=../aarch64-gnu.toolchain.cmake
  good_nproc
  jobs=$?
  make -j${jobs}
  make install
  cd -
}

build_ncnn() {
  if [ ! -e "ncnn" ];then
    git clone https://github.com/tencent/ncnn --branch 20220729 --depth=1
  fi
  if [ ! -e "ncnn/build_aarch64" ];then
    mkdir -p ncnn/build_aarch64
  fi
  cd ncnn/build_aarch64
  rm -rf CMakeCache.txt
  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=/tmp/ncnn-aarch64
  good_nproc
  jobs=$?
  make -j${jobs}
  make install
  cd -
}

build_mmdeploy() {
  git submodule init
  git submodule update

  if [ ! -e "build_aarch64" ];then
    mkdir build_aarch64
  fi
  cd build_aarch64

  rm -rf CMakeCache.txt
  cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/aarch64-linux-gnu.cmake \
    -DMMDEPLOY_TARGET_DEVICES="cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="ncnn" \
    -Dncnn_DIR=/tmp/ncnn-aarch64/lib/cmake/ncnn \
    -DOpenCV_DIR=/tmp/ocv-aarch64/lib/cmake/opencv4

  good_nproc
  jobs=$?
  make -j${jobs}
  make install

  ls -lah install/bin/*
}

print_success() {
  echo "----------------------------------------------------------------------"
  echo "Cross build finished, PLS copy bin/model/test_data to the device.. QVQ"
  echo "----------------------------------------------------------------------"
}

if [ ! -e "../mmdeploy-dep" ];then
  mkdir ../mmdeploy-dep
fi
cd ../mmdeploy-dep

install_tools
build_ocv
build_ncnn

cd ../mmdeploy
build_mmdeploy
print_success

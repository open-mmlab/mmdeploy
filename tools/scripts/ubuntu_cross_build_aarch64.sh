#!/bin/bash
set -ex
# get appropriate proc number: max(1, nproc-2)
good_nproc() {
  num=`nproc`
  num=`expr $num - 2`
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
    git clone https://github.com/opencv/opencv --depth=1 --branch=4.x --recursive
  fi
  if [ ! -e "opencv/platforms/linux/cross_build_aarch64" ];then
    mkdir opencv/platforms/linux/cross_build_aarch64
  fi
  cd opencv/platforms/linux/cross_build_aarch64
  rm -rf CmakeCache
  cmake -DCMAKE_INSTALL_PREFIX=/tmp/ocv-aarch64 -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake ../../..
  good_nproc
  jobs=$?
  echo "using  ${jobs}"
  make -j${jobs}
  make install
  cd -
}

build_ncnn() {
  echo ""
}

build_mmdeploy() {
  echo ""
}

if [ ! -e "mmdeploy-dep" ];then
  mkdir mmdeploy-dep
fi
cd mmdeploy-dep

install_tools
build_ocv
build_ncnn
build_mmdeploy

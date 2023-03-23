#!/bin/bash
set -e

opencvVer="4.5.5"

WORKSPACE=$(realpath $(dirname "$0"))
THIRDPARTY_DIR="${WORKSPACE}/thirdparty"

if [ ! -d $THIRDPARTY_DIR ]; then
    echo $THIRDPARTY_DIR
    mkdir -p $THIRDPARTY_DIR
fi

pushd ${THIRDPARTY_DIR}

url="https://github.com/opencv/opencv/archive/refs/tags/$opencvVer.tar.gz"
wget $url
tar xf $opencvVer.tar.gz
mv opencv-$opencvVer opencv

pushd opencv

mkdir build
pushd build
cmake .. -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=../install
make -j$(nproc)
make install

pushd -3

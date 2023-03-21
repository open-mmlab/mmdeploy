#!/bin/bash

WORKSPACE=$(realpath $(dirname "$0"))
OPENCV_DIR=""

if [ -n "$1" ]; then
    OPENCV_DIR=$(cd "$1"; pwd)
    if [ $? -ne 0 ]; then
        echo "opencv path $1 doesn't exist"
        exit 1
    fi
    if [ ! -f "$OPENCV_DIR/OpenCVConfig.cmake" ]; then
        echo "opencv path $1 doesn't contains OpenCVConfig.cmake"
        exit 1
    fi
fi

if [ -z "$OPENCV_DIR" ]; then
    # search thirdparty
    OPENCV_DIR="${WORKSPACE}/thirdparty/opencv/install/lib64/cmake/opencv4"
    _OPENCV_DIR="${WORKSPACE}/thirdparty/opencv/install/lib/cmake/opencv4"
    if [ -d "$OPENCV_DIR" ]; then
        echo "Found OPENCV_DIR= $OPENCV_DIR"
    elif [ -d "$_OPENCV_DIR" ]; then
        OPENCV_DIR=$_OPENCV_DIR
        echo "Found OPENCV_DIR= $OPENCV_DIR"
    else
        echo "Can't find opencv, please provide OPENCV_DIR or install it by install_opencv.sh"
        exit 1
    fi
fi

MMDEPLOY_DIR="$WORKSPACE/lib/cmake/MMDeploy"

BUILD_DIR="${WORKSPACE}/example/cpp/build"
if [ -d "${BUILD_DIR}" ]; then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}


cmake .. -DMMDeploy_DIR="$MMDEPLOY_DIR" \
            -DOpenCV_DIR="${OPENCV_DIR}"

make -j $(nproc)

cd ${WORKSPACE}
ln -sf ${BUILD_DIR} bin

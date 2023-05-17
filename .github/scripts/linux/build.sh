#!/bin/bash

ARGS=("$@")

SCRIPT_DIR=$(cd `dirname $0`; pwd)
MMDEPLOY_DIR=$SCRIPT_DIR/../../..

cd $MMDEPLOY_DIR
mkdir -p build && cd build
cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="$1" \
    -DMMDEPLOY_TARGET_BACKENDS="$2" "${ARGS[@]:2}"

make -j$(nproc) && make install

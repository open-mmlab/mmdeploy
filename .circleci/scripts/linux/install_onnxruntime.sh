#!/bin/bash

if [ $# != 2 ]; then
    echo "wrong command. usage: bash install_onnxruntime.sh <cpu|cuda> <version>"
    exit 1
fi

PLATFORM=$1
VERSION=$2

if [ "$PLATFORM" == 'cpu' ]; then
    python -m pip install onnxruntime=="$VERSION"

    wget https://github.com/microsoft/onnxruntime/releases/download/v"$VERSION"/onnxruntime-linux-x64-"$VERSION".tgz
    tar -zxvf onnxruntime-linux-x64-"$VERSION".tgz
    ln -sf onnxruntime-linux-x64-"$VERSION" onnxruntime
elif [ "$PLATFORM" == 'cuda' ]; then
    pip install onnxruntime-gpu=="$VERSION"

    wget https://github.com/microsoft/onnxruntime/releases/download/v"$VERSION"/onnxruntime-linux-x64-gpu-"$VERSION".tgz
    tar -zxvf onnxruntime-linux-x64-gpu-"$VERSION".tgz
    ln -sf onnxruntime-linux-x64-gpu-"$VERSION" onnxruntime
else
    echo "'$PLATFORM' is not supported"
    exit 1
fi

export ONNXRUNTIME_DIR=$(pwd)/onnxruntime
echo "export ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

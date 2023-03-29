#!/bin/bash

if [ -n "$1" ]; then
    WORKSPACE=$1
else
    WORKSPACE=$(realpath $(dirname "${BASH_SOURCE[0]}"))
fi

THIRDPARTY_DIR=$WORKSPACE/thirdparty

pushd $THIRDPARTY_DIR

if [ -d onnxruntime ]; then
    export ONNXRUNTIME_DIR=$THIRDPARTY_DIR/onnxruntime
    export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
fi

if [ -d tensorrt ]; then
    export TENSORRT_DIR=$THIRDPARTY_DIR/tensorrt
    export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
fi

if [ -d openvino ]; then
    export InferenceEngine_DIR=$THIRDPARTY_DIR/runtime/cmake
    sopaths=$(find $(pwd)/openvino -name "*.so"  -exec dirname {} \; | uniq | tr '\n' ':')
    export LD_LIBRARY_PATH=$sopaths$LD_LIBRARY_PATH
fi

popd

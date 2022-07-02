#!/bin/bash

if [ $# != 2 ]; then
    echo "wrong command. usage: bash converter.sh <codebase> <work dir>"
    exit 1
fi

if [ "$1" == 'mmcls' ]; then
    python3 -m pip install mmcls
    git clone --recursive https://github.com/open-mmlab/mmclassification.git
    wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
    python3 mmdeploy/tools/deploy.py \
            mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py \
            mmclassification/configs/resnet/resnet18_8xb32_in1k.py \
            resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
            mmclassification/demo/demo.JPEG \
            --work-dir "$2" --dump-info
fi

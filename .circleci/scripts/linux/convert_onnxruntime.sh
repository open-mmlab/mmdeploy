#!/bin/bash

if [ $# != 2 ]; then
    echo "wrong command. usage: bash converter.sh <codebase> <work dir>"
    exit 1
fi

if [ "$1" == 'mmcls' ]; then
    python3 -m mim install $(cat mmdeploy/requirements/codebases.txt | grep mmcls)
    python3 -m mim download mmcls --config resnet18_8xb32_in1k --dest .
    python3 mmdeploy/tools/deploy.py \
            mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py \
            ./resnet18_8xb32_in1k.py \
            ./resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
            mmdeploy/tests/data/tiger.jpeg \
            --work-dir "$2" --dump-info
fi

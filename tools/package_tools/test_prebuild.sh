#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.
# Script for test pre-build wheel

BUILD_CONFIG=$1

PARAMETER_NUMBER=1
if [ $# != ${PARAMETER_NUMBER} ]; then
  echo "Number of parameter got $#, which expected for ${PARAMETER_NUMBER} !!!"
  exit 1
fi

# build convert wheel
conda activate mmdeploy
python "${MMDEPLOY_DIR}/tools/package_tools/mmdeploy_builder.py" \
  "${MMDEPLOY_DIR}/tools/package_tools/configs/${BUILD_CONFIG}" \
  "${MMDEPLOY_DIR}"

# using another env to pip install it
c

# test convert with RetinaNet of mmdetection
mkdir /tmp/wheel_convert_test && cd /tmp/wheel_convert_test
export CONVERT_TEST_PATH=$(pwd)
wget https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth

cd "../${MMDEPLOY_DIR}" || exit 1
if [ ! -d mmdetection ]; then
  git clone -b master https://github.com/open-mmlab/mmdetection.git
fi

cd "${MMDEPLOY_DIR}" || exit 1
python tools/deploy.py \
  configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
  ../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py \
  ${CONVERT_TEST_PATH}/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \
  ../mmdetection/demo/demo.jpg \
  --device cuda \
  --work-dir ${CONVERT_TEST_PATH}/retinanet_output \
  --dump-info || exit 2

# build sdk wheel
python "${MMDEPLOY_DIR}/tools/package_tools/mmdeploy_builder.py" \
  "${MMDEPLOY_DIR}/tools/package_tools/configs/${BUILD_CONFIG}" \
  "${MMDEPLOY_DIR}"

cd ${MMDEPLOY_DIR}/build/install/example || exit 1
mkdir -p build && cd build
cmake .. -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy
make -j$(nproc)

# test sdk demo
conda deactivate
conda activate mmdeploy_sdk
pip uninstall mmdeploy_builder
pip install xxx.whl
./object_detection cuda \
  "${CONVERT_TEST_PATH}/retinanet_output" \
  "${MMDEPLOY_DIR}/../mmdetection/demo/demo.jpg"  || exit 3

exit 0

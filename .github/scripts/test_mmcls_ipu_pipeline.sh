#!/bin/sh

set -e
# print env
python3 tools/check_env.py

deploy_cfg=configs/mmcls/classification_ipu_static.py
device=cpu
model_cfg=../mmclassification/configs/resnet/resnet18_8xb32_in1k.py
checkpoint=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
input_img=../mmclassification/demo/demo.JPEG
work_dir=work_dir

echo "------------------------------------------------------------------------------------------------------------"
echo "deploy_cfg=$deploy_cfg"
echo "model_cfg=$model_cfg"
echo "checkpoint=$checkpoint"
echo "device=$device"
echo "------------------------------------------------------------------------------------------------------------"

mkdir -p $work_dir

python3 tools/deploy.py \
  $deploy_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir \
  --dump-info

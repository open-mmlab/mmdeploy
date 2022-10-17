#!/bin/sh

set -e
# print env
python3 tools/check_env.py

deploy_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
device=cpu
model_cfg=../mmclassification/configs/resnet/resnet18_8xb32_in1k.py
checkpoint=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
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

# prepare dataset
wget -P data/ https://github.com/open-mmlab/mmdeploy/files/9401216/imagenet-val100.zip
unzip data/imagenet-val100.zip -d data/

echo "Running test with ort"

python3 tools/test.py \
  $deploy_cfg \
  $model_cfg \
  --model $work_dir/end2end.onnx \
  --device $device \
  --out $work_dir/ort_out.pkl \
  --metrics accuracy \
  --device $device \
  --log2file $work_dir/test_ort.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 32

echo "Running test with sdk"

# change topk for test
sed -i 's/"topk": 5/"topk": 1000/g' work_dir/pipeline.json

python3 tools/test.py \
  $sdk_cfg \
  $model_cfg \
  --model $work_dir \
  --device $device \
  --out $work_dir/sdk_out.pkl \
  --metrics accuracy \
  --device $device \
  --log2file $work_dir/test_sdk.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 1

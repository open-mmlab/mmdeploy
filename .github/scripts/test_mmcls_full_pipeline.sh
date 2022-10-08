#!/bin/sh

set -e
# print env
python tools/check_env.py

deploy_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
device=cpu
python -m mim download mmcls --config resnet18_8xb32_in1k --dest ../
model_cfg=../resnet18_8xb32_in1k.py
checkpoint=../resnet18_8xb32_in1k_20210831-fbbb1da6.pth
sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
input_img=tests/data/tiger.jpeg
work_dir=work_dir

echo "------------------------------------------------------------------------------------------------------------"
echo "deploy_cfg=$deploy_cfg"
echo "model_cfg=$model_cfg"
echo "checkpoint=$checkpoint"
echo "device=$device"
echo "------------------------------------------------------------------------------------------------------------"

mkdir -p $work_dir

python tools/deploy.py \
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

python tools/test.py \
  $deploy_cfg \
  $model_cfg \
  --model $work_dir/end2end.onnx \
  --device $device \
  --device $device \
  --log2file $work_dir/test_ort.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 32

echo "Running test with sdk"

# change topk for test
sed -i 's/"topk": 5/"topk": 1000/g' work_dir/pipeline.json

python tools/test.py \
  $sdk_cfg \
  $model_cfg \
  --model $work_dir \
  --device $device \
  --device $device \
  --log2file $work_dir/test_sdk.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 1

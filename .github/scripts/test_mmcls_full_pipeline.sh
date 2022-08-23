#!/bin/bash

#resnet18 + cifar10

checkpoint=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
model_cfg=../mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
ort_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
input_img=../mmclassification/demo/demo.JPEG
work_dir=work_dir

mkdir -p $work_dir
device=cpu

python3 tools/deploy.py \
  $ort_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir \
  --dump-info

# prepare dataset
mkdir -p data/cifar10
#wget -P data/cifar10/ https://raw.githubusercontent.com/RunningLeon/mmdeploy-testdata/master/data/cifar10/cifar-10-python.tar.gz
#tar -xvf data/cifar10/cifar-10-python.tar.gz -C data/cifar10/
## change to avoid md5 check
#sed -i "s/get_dist_info()/1,0/g" ../mmclassification/mmcls/datasets/cifar.py

echo "Running test with ort"
python3 tools/test.py \
  $ort_cfg \
  $model_cfg \
  --model $work_dir/end2end.onnx \
  --device $device \
  --out $work_dir/out.pkl \
  --metrics accuracy \
  --device $device \
  --log2file $work_dir/test_ort.log \
  --speed-test \
  --log-interval 100 \
  --warmup 100 \
  --batch-size 64

echo "Running test with sdk"
python3 tools/test.py \
  $sdk_cfg \
  $model_cfg \
  --model $work_dir \
  --device $device \
  --out $work_dir/out.pkl \
  --metrics accuracy \
  --device $device \
  --log2file $work_dir/test_sdk.log \
  --speed-test \
  --log-interval 100 \
  --warmup 100 \
  --batch-size 64

#!/bin/bash

mmcls_version=$1

echo  "$mmcls_version"

git clone  --depth 1 --single-branch --branch $mmcls_version git@github.com:open-mmlab/mmclassification.git ../mmclassification

checkpoint=https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
model_cfg=../mmclassification/configs/resnet/resnet18_8xb16_cifar10.py
ort_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
input_img=../mmclassification/demo/demo.JPEG
work_dir=work_dir

mkdir -p $work_dir
device=cpu

python tools/deploy.py \
  $ort_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir \
  --dump-info

mkdir -p data/cifar10

wget -P data/cifar10 https://raw.githubusercontent.com/RunningLeon/mmdeploy-testdata/master/data/cifar10/cifar-10-python.tar.gz

# change md5
old_md5=c58f30108f718f92721af3b95e74349a
new_md5=7a7c9263ffcde45dd229228f7dd02f5a
sed -i "s/$old_md5/$new_md5/g" ../mmclassification/mmcls/datasets/cifar.py

python tools/test.py \
  $ort_cfg \
  $model_cfg \
  --model $work_dir/end2end.onnx \
  --device $device \
  --out $work_dir/out.pkl \
  --metrics accuracy \
  --device $device \
  --log2file $work_dir/test.log \
  --speed-test \
  --log-interval 1 \
  --warmup 2 \
  --batch-size 64

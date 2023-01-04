#!/bin/sh
set -e

export bash_dir=$(cd `dirname $0`; pwd)
export MMDEPLOY_DIR=$bash_dir/../../..
cd $MMDEPLOY_DIR

# prepare dataset
wget -P data/ https://github.com/open-mmlab/mmdeploy/files/9401216/imagenet-val100.zip
unzip data/imagenet-val100.zip -d data/


# check env
echo "========================  check_env.py  ========================"
python3 tools/check_env.py

deploy_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
device=cpu

sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
input_img=tests/data/tiger.jpeg
work_dir=work_dir
mkdir -p $work_dir
python3 -m mim download mmcls --config resnet18_8xb32_in1k --dest $work_dir
model_cfg=$work_dir/resnet18_8xb32_in1k.py
checkpoint=$work_dir/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

echo "------------------------------------------------------------------------------------------------------------"
echo "deploy_cfg=$deploy_cfg"
echo "model_cfg=$model_cfg"
echo "checkpoint=$checkpoint"
echo "device=$device"
echo "------------------------------------------------------------------------------------------------------------"

echo "========================  deploy.py  ========================"
python3 tools/deploy.py \
  $deploy_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir \
  --dump-info

echo "========================  torch2onnx.py  ========================"
python3 tools/torch2onnx.py \
  $deploy_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir/torch2onnx

ls $work_dir/torch2onnx

echo "========================  profiler.py  ========================"
# onnxruntime model
python3 tools/profiler.py \
  $deploy_cfg \
  $model_cfg \
  ./data \
  --model $work_dir/end2end.onnx \
  --device $device \
  --shape 224x224 \
  --batch-size 1

# pytorch model
python3 tools/profiler.py \
  $deploy_cfg \
  $model_cfg \
  ./data \
  --model $checkpoint \
  --device $device \
  --shape 224x224 \
  --batch-size 32

echo "Running test with ort"
echo "========================  test.py + ort  ========================"
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

echo "====================="
echo "Running test with sdk"

# change topk for test
sed -i 's/"topk": 5/"topk": 1000/g' work_dir/pipeline.json

echo "========================  test.py + sdk ========================"
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

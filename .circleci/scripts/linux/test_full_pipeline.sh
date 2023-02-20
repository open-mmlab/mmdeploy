#!/bin/sh

set -e
# print env
#python3 tools/check_env.py
backend=${1:-ort}
device=${2:-cpu}
current_dir=$(cd `dirname $0`; pwd)
mmdeploy_dir=$current_dir/../../../
cd $mmdeploy_dir

model_cfg=../resnet18_8xb32_in1k.py
checkpoint=../resnet18_8xb32_in1k_20210831-fbbb1da6.pth
sdk_cfg=configs/mmcls/classification_sdk_dynamic.py
input_img=tests/data/tiger.jpeg
work_dir=work_dir
mkdir -p $work_dir data

python3 -m mim download mmcls --config resnet18_8xb32_in1k --dest $work_dir

if [ $backend == "ort" ]; then
    deploy_cfg=configs/mmcls/classification_onnxruntime_dynamic.py
    model=$work_dir/end2end.onnx
elif [ $backend == "trt" ]; then
    deploy_cfg=configs/mmcls/classification_tensorrt-fp16_dynamic-224x224-224x224.py
    model=$work_dir/end2end.engine
else
  echo "Unsupported Backend=$backend"
  exit
fi

echo "------------------------------------------------------------------------------------------------------------"
echo "deploy_cfg=$deploy_cfg"
echo "model_cfg=$model_cfg"
echo "checkpoint=$checkpoint"
echo "device=$device"
echo "------------------------------------------------------------------------------------------------------------"

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

echo "Running test with $backend"

python3 tools/test.py \
  $deploy_cfg \
  $model_cfg \
  --model $model \
  --device $device \
  --log2file $work_dir/test_ort.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 8

echo "Running test with sdk"

# change topk for test
sed -i 's/"topk": 5/"topk": 1000/g' work_dir/pipeline.json

python3 tools/test.py \
  $sdk_cfg \
  $model_cfg \
  --model $work_dir \
  --device $device \
  --log2file $work_dir/test_sdk.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 8

# test profiler
echo "Profile sdk model"
python3 tools/profiler.py \
  $sdk_cfg \
  $model_cfg \
  ./data \
  --model $work_dir \
  --device $device \
  --batch-size 8 \
  --shape 224x224

rm -rf $work_dir $pwd/data
echo "All done"

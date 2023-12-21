#!/bin/sh

set -e
# print env
#python3 tools/check_env.py
backend=${1:-ort}
device=${2:-cpu}
current_dir=$(cd `dirname $0`; pwd)
mmdeploy_dir=$current_dir/../../..
cd $mmdeploy_dir

work_dir=$mmdeploy_dir/work_dir
mkdir -p $work_dir $mmdeploy_dir/data

model_cfg=$work_dir/resnet18_8xb32_in1k.py
checkpoint=$work_dir/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
sdk_cfg=configs/mmpretrain/classification_sdk_dynamic.py
input_img=tests/data/tiger.jpeg

python3 -m mim download mmpretrain --config resnet18_8xb32_in1k --dest $work_dir

if [ $backend == "ort" ]; then
    deploy_cfg=configs/mmpretrain/classification_onnxruntime_dynamic.py
    model=$work_dir/end2end.onnx
elif [ $backend == "trt" ]; then
    deploy_cfg=configs/mmpretrain/classification_tensorrt-fp16_dynamic-224x224-224x224.py
    model=$work_dir/end2end.engine
elif [ $backend == "ncnn" ]; then
    deploy_cfg=configs/mmpretrain/classification_ncnn_static.py
    model="$work_dir/end2end.param $work_dir/end2end.bin"
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

if [ $backend == "trt" ]; then
    echo "Running onnx2tensorrt"
    python3 tools/onnx2tensorrt.py \
    $deploy_cfg \
    $work_dir/end2end.onnx \
    $work_dir/temp
fi

# prepare dataset
wget -P data/ https://github.com/open-mmlab/mmdeploy/releases/download/v0.1.0/imagenet-val100.zip
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

echo "All done"

#!/bin/sh

set -e

ip=${1}
port=${2:8585}

# create http server
nohup python3 -m http.server --directory /data2/shared $port > tmp.log 2>&1

export TENSORRT_URL=http://$ip:$port/TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
export TENSORRT_VERSION=8.2.3.0
export CUDA_INT=113

# build docker image
docker build ./docker/Base/ -t runningleon814/mmdeploy-base:ubuntu20.04-cuda11.3 \
    --build-arg CUDA_INT=$CUDA_INT \
    --build-arg TENSORRT_URL=${TENSORRT_URL} \
    --build-arg TENSORRT_VERSION=${TENSORRT_VERSION}

# test docker image
docker run --gpus=all -itd \
  -v /data2/benchmark:/root/workspace/openmmlab-data \
  -v /data2/checkpoints:/root/workspace/mmdeploy_checkpoints \
  -v ~/mmdeploy:/root/workspace/mmdeploy \
  --name test_mmdeploy_base \
  runningleon814/mmdeploy-base:20.04-cu11.3

# push to docker hub
docker login
docker push runningleon814/mmdeploy-base:20.04-cu11.3

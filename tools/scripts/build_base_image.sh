#!/bin/sh

set -e

ip=${1}
port=${2:8585}

date_today=`date +'%Y%m%d'`

# create http server
nohup python3 -m http.server --directory /data2/shared/nvidia $port > tmp.log 2>&1

export TENSORRT_URL=http://$ip:$port/TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
export TENSORRT_VERSION=8.2.3.0
export CUDA_INT=113
export TAG=ubuntu20.04-cuda11.3

# build docker image
docker build ./docker/Base/ -t openmmlab/mmdeploy:$TAG \
    --build-arg CUDA_INT=$CUDA_INT \
    --build-arg TENSORRT_URL=${TENSORRT_URL} \
    --build-arg TENSORRT_VERSION=${TENSORRT_VERSION}

docker tag openmmlab/mmdeploy:$TAG openmmlab/mmdeploy:${TAG}-${date_today}

# test docker image
docker run --gpus=all -itd \
  -v /data2/benchmark:/root/workspace/openmmlab-data \
  -v /data2/checkpoints:/root/workspace/mmdeploy_checkpoints \
  -v ~/mmdeploy:/root/workspace/mmdeploy \
  openmmlab/mmdeploy:$TAG


# push to docker hub
docker login
docker push openmmlab/mmdeploy:$TAG
docker push openmmlab/mmdeploy:$TAG-${date_today}

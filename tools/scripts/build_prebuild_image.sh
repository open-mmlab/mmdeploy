#!/bin/sh

set -e

ip=${1}
port=${2:8585}

date_today=`date +'%Y%m%d'`

# create http server
nohup python3 -m http.server --directory /data2/shared/mmdeploy-manylinux2014_x86_64-cuda11.3 $port > tmp.log 2>&1

export ip=10.1.52.36
export port=8585
export CUDA_URL=http://$ip:$port/cuda_11.3.0_465.19.01_linux.run
export CUDNN_URL=http://$ip:$port/cudnn-11.3-linux-x64-v8.2.1.32.tgz
export TENSORRT_URL=http://$ip:$port/TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
export TENSORRT_VERSION=8.2.3.0
export TAG=manylinux2014_x86_64-cuda11.3

# build docker image
docker build ./docker/prebuild/ -t openmmlab/mmdeploy:$TAG \
    --build-arg CUDA_URL=$CUDA_URL \
    --build-arg CUDNN_URL=$CUDNN_URL \
    --build-arg TENSORRT_URL=${TENSORRT_URL}

# push to docker hub
docker login
docker push openmmlab/mmdeploy:$TAG

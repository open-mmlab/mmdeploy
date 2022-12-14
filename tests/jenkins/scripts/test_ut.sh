#!/bin/bash

set -e

## parameters
config_filename=${1:-default.config}
config="${HOME}/mmdeploy/tests/jenkins/conf/${config_filename}"
if [ -f "$config" ]; then
    echo "Using config: $config"
else
    echo "$config does not exist."
    exit 1
fi

echo "config==================="
echo $config
echo "========================="

docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')
repo_version=$(grep repo_version ${config} | sed 's/repo_version=//')
tensorrt_version=$(grep tensorrt_version ${config} | sed 's/tensorrt_version=//')
cuda_version=$(echo $docker_image | awk '{split($0,a,"-"); print a[5]}')

trt_dir=/data2/shared/nvidia-packages/TensorRT-${tensorrt_version}-${cuda_version}
if [ -d "$trt_dir" ]; then
    echo "TensorRT directory $trt_dir"
else
    echo "$trt_dir not exist."
    exit 1
fi

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/ut_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}
chmod 777 ${log_dir}/../

container_name=openmmlab${repo_version}-ut-${time_snap}
container_id=$(
    docker run -itd \
        --gpus all \
        --ipc=host \
        -v ${log_dir}:/root/workspace/ut_log \
        -v ${HOME}/mmdeploy/tests/jenkins:/root/workspace/jenkins \
        -v ${trt_dir}:/root/workspace/TensorRT \
        --name ${container_name} \
        ${docker_image} /bin/bash
)
echo "container_id=${container_id}"
nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} && \
/root/workspace/jenkins/scripts/docker_exec_ut.sh ${config_filename}" >${log_dir}/test_ut.log 2>&1 &
wait
docker stop $container_id

cat ${log_dir}/test_ut.log

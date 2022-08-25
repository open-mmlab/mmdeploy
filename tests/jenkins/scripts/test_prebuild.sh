#!/bin/bash

## parameters
export docker_image=$1

log_dir=/data2/regression_log/prebuild/$(date +%Y%m%d)/$(date +%Y%m%d%H%M)
mkdir -p ${log_dir}

container_name=prebuild-$(date +%Y%m%d%H%M)
container_id=$(
    docker run \
        --gpus all \
        -itd \
        -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
        -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
        -v /data2/shared/prebuilt-mmdeploy:/root/workspace/prebuild-mmdeploy \
        -v ${log_dir}:/root/workspace/log \
        --name ${container_name} \
        ${docker_image} /bin/bash
    ) 
echo "container_id=${container_id}"
nohup docker exec ${container_id} bash -c "git clone --recursive https://github.com/kumailf/mmdeploy.git && /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_prebuild.sh" > ${log_dir}/prebuild.log 2>&1 & 
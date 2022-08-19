#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)

## build

docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}

# docker run cmd for convert
for codebase in ${codebase_list[@]}
do
    log_dir=/data2/regression_log/$(date +%Y%m%d)/$(date +%Y%m%d%H%M)
    mkdir -p ${log_dir}
    container_id=$(
        docker run -itd \
            -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
            -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
            -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
            ${docker_image} /bin/bash
    )
    docker exec ${container_id} git clone --recursive https://github.com/kumailf/mmdeploy.git
    docker exec ${container_id} bash -c "/root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_convert.sh ${codebase}"
done
#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)

# docker run cmd for convert
for codebase in ${codebase_list[@]}
do
    log_dir=/data2/regression_log/$(date +%Y%m%d)/$(date +%Y%m%d%H%M)
    mkdir -p ${log_dir}
    container_id=$(
        docker run -itd \
            --gpus all \
            -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
            -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
            -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
            ${docker_image} /bin/bash
    )
    nohup docker exec ${container_id} bash -c "git clone --recursive https://github.com/kumailf/mmdeploy.git && /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_convert_gpu.sh ${codebase}" > ${codebase}.log 2>&1 &
done
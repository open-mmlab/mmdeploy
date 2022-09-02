#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)

# docker run cmd for convert
for codebase in ${codebase_list[@]}
do
    log_dir=/data2/regression_log/
    mkdir -p ${log_dir}
    container_name=convert-${codebase}-$(date +%Y%m%d%H%M)
    container_id=$(
        docker run -itd \
            --gpus all \
            -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
            -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
            -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
            -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
            --name ${container_name} \
            ${docker_image} /bin/bash
    )
    echo "container_id=${container_id}"
    nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch master --recursive https://github.com/open-mmlab/mmdeploy.git &&\
     /root/workspace/mmdeploy_script/docker_exec_convert_gpu.sh ${codebase}" > ${log_dir}/${codebase}.log 2>&1 &
done

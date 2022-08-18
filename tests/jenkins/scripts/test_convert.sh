#!/bin/bash

## parameters

export docker_image=$1
export codebase_list=($2)

## build

docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}
export container_id=$(docker run -itd ${docker_image} /bin/bash) 


# docker run cmd for convert
for codebase in ${codebase_list}
do
    log_dir=/data2/regression_log/$(date +%Y%m%d%H%M)/${codebase}
    mkdir -p ${log_dir}
    container_id=$(
        docker run -it \
            -v /data2/checkpoints/${codebase}:/root/workspace/mmdeploy_checkpoints \
            -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
            -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
            --name ${codebase}-${docker_image} \
            ${docker_image}
    )
    docker exec ${container_id} git clone https://github.com/kumailf/mmdeploy.git
    docker exec ${container_id} bash -c "/root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_for_build.sh ${codebase}"
done
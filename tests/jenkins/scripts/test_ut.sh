#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)

date_snap=$(date +%Y%m%d%H%M)
# docker run cmd for convert
for codebase in ${codebase_list[@]}
do
    log_dir=/data2/ut_log/${date_snap}
    mkdir -p ${log_dir}
    container_name=ut-${codebase}-${date_snap}
    container_id=$(
        docker run -itd \
            --gpus all \
            -v ${log_dir}:/root/workspace/ut_log \
            --name ${container_name} \
            ${docker_image} /bin/bash
    )
    echo "container_id=${container_id}"
    nohup docker exec ${container_id} bash -c "git clone --recursive https://github.com/kumailf/mmdeploy.git && /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_ut.sh ${codebase}" > ${log_dir}/${codebase}.log 2>&1 &
done

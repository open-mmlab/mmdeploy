#!/bin/bash

## parameters

docker_image=$1
codebase_list=($2)

odate_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
# docker run cmd for unittest
for codebase in ${codebase_list[@]}
do
    log_dir=/data2/regression_log/ut_log/${date_snap}/${time_snap}
    mkdir -p ${log_dir}
    container_name=ut-${codebase}-${time_snap}
    container_id=$(
        docker run -itd \
            --gpus all \
            -v ${log_dir}:/root/workspace/ut_log \
            -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
            --name ${container_name} \
            ${docker_image} /bin/bash
    )
    echo "container_id=${container_id}"
    nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch master --recursive https://github.com/open-mmlab/mmdeploy.git &&\
     /root/workspace/mmdeploy_script/docker_exec_ut.sh ${codebase}" > ${log_dir}/${codebase}.log 2>&1 &
done

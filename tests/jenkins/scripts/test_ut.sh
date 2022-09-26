#!/bin/bash

## parameters
config="tests/jenkins/${1:-default.config}"

docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
codebase_list=($(grep codebase_list ${config} | sed 's/codebase_list=//'))
exec_performance=$(grep exec_performance ${config} | sed 's/exec_performance=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/ut_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}

## docker run cmd for unittest
for codebase in ${codebase_list[@]}; do

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
    nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} && \
    /root/workspace/mmdeploy_script/docker_exec_ut.sh ${codebase}" >${log_dir}/${codebase}.log 2>&1 &
    echo "${codebase} unittest finish!"
    cat ${log_dir}/${codebase}.log
done
wait

#!/bin/bash

## parameters
config="${HOME}/mmdeploy/tests/jenkins/conf/${1:-default.config}"
if [ -f "$config" ]; then
    echo "Using config: $config"
else
    echo "$config does not exist."
    exit 1
fi

docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
codebase_list=($(grep codebase_list ${config} | sed 's/codebase_list=//'))
exec_performance=$(grep exec_performance ${config} | sed 's/exec_performance=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')
repo_version=$(grep repo_version ${config} | sed 's/repo_version=//')
REQUIREMENT=$(grep requirement ${config} | sed 's/requirement=//')

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/ut_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}
chmod 777 ${log_dir}/../

## docker run cmd for unittest
for codebase in ${codebase_list[@]}; do

    container_name=openmmlab${repo_version}-ut-${codebase}-${time_snap}
    container_id=$(
        docker run -itd \
            --gpus all \
            -v ${log_dir}:/root/workspace/ut_log \
            -v ~/mmdeploy/tests/jenkins:/root/workspace/jenkins\
            --name ${container_name} \
            ${docker_image} /bin/bash
    )
    echo "container_id=${container_id}"
    nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} && \
    cp -R /root/workspace/jenkins/ mmdeploy/tests/ && \
    /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_ut.sh ${codebase} ${REQUIREMENT}" >${log_dir}/${codebase}.log 2>&1 &
done
wait

for codebase in ${codebase_list[@]}; do
    cat ${log_dir}/${codebase}.log

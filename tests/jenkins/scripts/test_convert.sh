#!/bin/bash

## parameters
config="/home/zhengshaofeng/mmdeploy/tests/jenkins/conf/${1:-default.config}"

docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
codebase_list=($(grep codebase_list ${config} | sed 's/codebase_list=//'))
exec_performance=$(grep exec_performance ${config} | sed 's/exec_performance=//')
max_job_nums=$(grep max_job_nums ${config} | sed 's/max_job_nums=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')
TENSORRT_VERSION=$(grep tensorrt_version ${config} | sed 's/tensorrt_version=//')
REQUIREMENT=$(grep requirement ${config} | sed 's/requirement=//')

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/convert_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}

## make & init mutex
trap "exec 1000>&-; exec 1000<&-;exit 0" 2
mkfifo mutexfifo
exec 1000<>mutexfifo
rm -rf mutexfifo
for ((n = 1; n <= ${max_job_nums}; n++)); do
    echo >&1000
done

## docker run cmd for convert
for codebase in ${codebase_list[@]}; do
    read -u1000
    {
        container_name=convert-${codebase}-${time_snap}
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
        nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
        /root/workspace/mmdeploy_script/docker_exec_convert_gpu.sh ${codebase} "${exec_performance}" ${TENSORRT_VERSION} ${REQUIREMENT}" >${log_dir}/${codebase}.log 2>&1 &
        wait
        docker stop $container_id
        echo "${codebase} convert finish!"
        cat ${log_dir}/${codebase}.log
        echo >&1000
    } &
done

wait
exec 1000>&-
exec 1000<&-

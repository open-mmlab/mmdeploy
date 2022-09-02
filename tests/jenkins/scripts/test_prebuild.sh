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
        -v /data2/checkpoints:/root/workspace/mmdeploy_checkpoints \
        -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
        -v /data2/shared/prebuilt-mmdeploy/$docker_image/$(date +%Y%m%d):/root/workspace/prebuild-mmdeploy \
        -v ${log_dir}:/root/workspace/log \
        -v /data2/shared/scripts:/root/workspace/mmdeploy_script \
        --name ${container_name} \
        ${docker_image} /bin/bash
    )
echo "container_id=${container_id}"
nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch master --recursive https://github.com/open-mmlab/mmdeploy.git &&\
 /root/workspace/mmdeploy_script/docker_exec_prebuild.sh" > ${log_dir}/prebuild.log 2>&1 &

#!/bin/bash

## parameters
export docker_image=$1

log_dir=/data2/regression_log/prebuild/$(date +%Y%m%d)/$(date +%Y%m%d%H%M).log

container_id=$(
    docker run \
        -itd \
        -v /data2/shared/prebuilt-mmdeploy:/root/workspace/prebuild-mmdeploy \
        ${docker_image} /bin/bash
    ) 
nohup docker exec ${container_id} bash -c "git clone --recursive https://github.com/kumailf/mmdeploy.git && /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_prebuild.sh" > ${log_dir} 2>&1 &
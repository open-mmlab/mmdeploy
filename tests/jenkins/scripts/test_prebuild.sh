#!/bin/bash

## parameters
export docker_image=$1

container_id=$(
    docker run \
        -itd \
        -v /data2/shared/prebuilt-mmdeploy:/root/workspace/prebuild-mmdeploy \
        ${docker_image} /bin/bash
    ) 
nohup docker exec ${container_id} bash -c "git clone --recursive https://github.com/kumailf/mmdeploy.git && /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_prebuild.sh" > /data2/regression_log/prebuild/$(date +%Y%m%d)/$(date +%Y%m%d%H%M) 2>&1 &
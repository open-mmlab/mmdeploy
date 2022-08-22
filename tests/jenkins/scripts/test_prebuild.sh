#!/bin/bash

## parameters
export docker_image=$1

## build

docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}
container_id=$(
    docker run \
        -itd \
        -v /data2/shared/prebuilt-mmdeploy:/root/workspace/prebuild-mmdeploy \
        ${docker_image} /bin/bash
    ) 
docker exec -d ${container_id} git clone --recursive https://github.com/kumailf/mmdeploy.git
docker exec -d ${container_id} bash -c "/root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_prebuild.sh"
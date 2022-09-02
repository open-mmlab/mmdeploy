#!/bin/bash

## parameters
# export docker_image=mmdeploy-ci-ubuntu-18.04
export docker_image=$1

#TODO update to output log and change script path
container_id=$(docker run -itd --gpus all ${docker_image} /bin/bash)
container_name=build-$(date +%Y%m%d%H%M)
echo "container_id=${container_id} --name ${container_name}"
docker exec -d ${container_id} git clone --depth 1 --branch master --recursive https://github.com/open-mmlab/mmdeploy.git
docker exec -d ${container_id} bash -c "/root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_build.sh"

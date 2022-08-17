#!/bin/bash

export docker_image=mmdeploy-ci-ubuntu-18.04
docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}
export container_id=$(docker run -itd ${docker_image} /bin/bash) 
docker exec ${container_id} git clone https://github.com/kumailf/mmdeploy.git
docker exec ${container_id} cd /root/workspace/mmdeploy && git checkout jenkins
docker exec ${container_id} sh /root/workspace/mmdeploy/tests/jenkins/scripts/docker_exec_for_build.sh



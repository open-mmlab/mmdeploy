#!/bin/bash

docker_image=mmdeploy-ci-ubuntu-18.04
docker build tests/jenkins/docker/${docker_image}/ -t ${docker_image}
container_id=$(docker run -td ${docker_image} /bin/bash) 
docker exec ${container_id} "sh /root/workspace/scripts/docker_exec_for_build.sh"


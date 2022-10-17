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
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/build_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}

## docker run cmd for build
container_name=build-${time_snap}
container_id=$(
    docker run -itd \
        --gpus all \
        -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
        --name ${container_name} \
        ${docker_image} /bin/bash
)
echo "container_id=${container_id} --name ${container_name}"

nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
/root/workspace/mmdeploy_script/docker_exec_build.sh" >${log_dir}/build.log 2>&1 &

wait
docker stop $container_id

cat ${log_dir}/build.log

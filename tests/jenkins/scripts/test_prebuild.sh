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
repo_version=$(grep repo_version ${config} | sed 's/repo_version=//')
## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/prebuild_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}
chmod -R 777 ${log_dir}/../
prebuilt_dir=/data2/shared/prebuilt-mmdeploy/${docker_image}/${date_snap}/${time_snap}
mkdir -p -m 777 ${prebuilt_dir}
container_name=convert-${codebase}-${time_snap}

container_name=openmmlab${repo_version}-prebuild-$(date +%Y%m%d%H%M)
container_id=$(
    docker run \
        --gpus all \
        -itd \
        -v /data2/checkpoints:/root/workspace/mmdeploy_checkpoints \
        -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
        -v ${prebuilt_dir}:/root/workspace/prebuild-mmdeploy \
        -v ${log_dir}:/root/workspace/log \
        -v ~/mmdeploy/tests/jenkins/scripts:/root/workspace/mmdeploy_script \
        --name ${container_name} \
        ${docker_image} /bin/bash
)
echo "container_id=${container_id}"
nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
 /root/workspace/mmdeploy_script/docker_exec_prebuild.sh" >${log_dir}/prebuild.log 2>&1 &
wait
docker stop $container_id
cat ${log_dir}/prebuild.log

#!/bin/bash

set -e

## parameters
config="${HOME}/mmdeploy/tests/jenkins/conf/${1:-default.config}"
if [ -f "$config" ]; then
    echo "Using config: $config"
else
    echo "$config does not exist."
    exit 1
fi

echo "config==================="
echo $config
echo "========================="

docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')
repo_version=$(grep repo_version ${config} | sed 's/repo_version=//')
cuda_version=$(echo $docker_image | awk '{split($0,a,"-"); print a[5]}')
tensorrt_version=$(grep tensorrt_version ${config} | sed 's/tensorrt_version=//')

## make
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
time_start=$(date +%s)

prebuild_archive_dir=/data2/shared/prebuilt-mmdeploy/${docker_image}/${date_snap}/${time_snap}
prebuild_log=/data2/regression_log/prebuild_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${prebuild_log} ${prebuild_archive_dir}
chmod 777 ${prebuild_log}/.. ${prebuild_archive_dir}/..

log_file=$prebuild_log/exec_prebuild_log.txt

## get log_url
host_ip=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"|grep 10)
log_url=${host_ip}:8989/${prebuild_log/\/data2\/regression_log\//}

# decide tensorrt version
# install tensorrt
trt_dir=/data2/shared/nvidia-packages/TensorRT-${tensorrt_version}-${cuda_version}
if [ -d "$trt_dir" ]; then
    echo "TensorRT directory $trt_dir"
else
    echo "$trt_dir not exist."
    exit 1
fi

container_name=openmmlab${repo_version}-prebuild-cuda${cuda_version}-$(date +%Y%m%d%H%M)
container_id=$(
    docker run \
        --gpus all \
        --ipc=host \
        -itd \
        -v ${trt_dir}:/root/workspace/TensorRT \
        -v /data2/checkpoints:/root/workspace/mmdeploy_checkpoints \
        -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
        -v ${prebuild_log}:/root/workspace/prebuild-mmdeploy \
        -v ${HOME}/mmdeploy/tests/jenkins:/root/workspace/jenkins \
        --name ${container_name} \
        ${docker_image} /bin/bash
)
echo "container_id=${container_id}"
nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
 /root/workspace/jenkins/scripts/docker_exec_prebuild.sh ${repo_version}" > ${log_file} 2>&1 &
wait
docker stop $container_id
cp -rf $prebuild_log/* $prebuild_archive_dir/

echo "查看日志: ${log_file}"

echo "end_time-$(date +%Y%m%d%H%M)"
time_end=$(date +%s)
take=$(( time_end - time_start ))
echo Time taken to execute commands is ${take} seconds.

#!/bin/bash

set -e

## parameters
config_filename=${1:-default.config}
config="${HOME}/mmdeploy/tests/jenkins/conf/${config_filename}"
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
codebase_list=($(grep codebase_list ${config} | sed 's/codebase_list=//'))
max_job_nums=$(grep max_job_nums ${config} | sed 's/max_job_nums=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')
repo_version=$(grep repo_version ${config} | sed 's/repo_version=//')
tensorrt_version=$(grep tensorrt_version ${config} | sed 's/tensorrt_version=//')
cuda_version=$(echo $docker_image | awk '{split($0,a,"-"); print a[5]}')

# check trt exists
trt_dir=/data2/shared/nvidia-packages/TensorRT-${tensorrt_version}-${cuda_version}
if [ -d "$trt_dir" ]; then
    echo "TensorRT directory $trt_dir"
else
    echo "$trt_dir not exist."
    exit 1
fi

## make log_dir
date_snap=$(date +%Y%m%d)
time_snap=$(date +%Y%m%d%H%M)
log_dir=/data2/regression_log/convert_log/${date_snap}/${time_snap}
mkdir -p -m 777 ${log_dir}
chmod 777 ${log_dir}/../

## get log_url
host_ip=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"|grep 10)
log_url=${host_ip}:8989/${log_dir/\/data2\/regression_log\//}

# let container know the host log path and url prefix
echo ${log_dir} > $log_dir/log_path.cfg
cp -f /data2/regression_log/host.cfg $log_dir/

## make & init mutex
trap "exec 1000>&-; exec 1000<&-;exit 0" 2
mkfifo mutexfifo
exec 1000<>mutexfifo
rm -rf mutexfifo
for ((n = 1; n <= ${max_job_nums}; n++)); do
    echo >&1000
done

## docker run cmd for convert
for codebase in ${codebase_list[@]}; do
    read -u1000
    {
        container_name=openmmlab${repo_version}-convert-${codebase}-${time_snap}
        container_id=$(
            docker run -itd \
                --ipc=host \
                --gpus all \
                -v /data2/checkpoints/:/root/workspace/mmdeploy_checkpoints \
                -v ${log_dir}:/root/workspace/mmdeploy_regression_working_dir \
                -v /data2/benchmark:/root/workspace/mmdeploy_benchmark \
                -v ${HOME}/mmdeploy/tests/jenkins:/root/workspace/jenkins\
                -v ${trt_dir}:/root/workspace/TensorRT \
                --name ${container_name} \
                ${docker_image} /bin/bash
        )
        nohup docker exec ${container_id} bash -c "git clone --depth 1 --branch ${mmdeploy_branch} --recursive ${repo_url} &&\
        /root/workspace/jenkins/scripts/docker_exec_convert_gpu.sh ${codebase} ${config_filename}" >${log_dir}/${codebase}_log.txt 2>&1 &
        wait
        docker stop $container_id
        echo "${codebase} convert finish!"
        echo "container_id=${container_id}"
        # cat ${log_dir}/${codebase}_log.txt
        echo "查看日志: ${log_url}/${codebase}_log.txt"
        echo "查看任务运行结果: ${log_url}/${codebase}/ \n"

        echo >&1000
    } &
done

wait
exec 1000>&-
exec 1000<&-

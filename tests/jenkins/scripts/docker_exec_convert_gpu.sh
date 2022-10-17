#!/bin/bash
start=$(date +%s)
## keep container alive
nohup sleep infinity >sleep.log 2>&1 &

## init conda
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup

## functions
function getFullName() {
    local codebase_=$1
    codebase_fullname=""
    if [ "$codebase_" = "mmdet" ]; then codebase_fullname="mmdetection"; fi
    if [ "$codebase_" = "mmcls" ]; then codebase_fullname="mmclassification"; fi
    if [ "$codebase_" = "mmdet3d" ]; then codebase_fullname="mmdetection3d"; fi
    if [ "$codebase_" = "mmedit" ]; then codebase_fullname="mmediting"; fi
    if [ "$codebase_" = "mmocr" ]; then codebase_fullname="mmocr"; fi
    if [ "$codebase_" = "mmpose" ]; then codebase_fullname="mmpose"; fi
    if [ "$codebase_" = "mmrotate" ]; then codebase_fullname="mmrotate"; fi
    if [ "$codebase_" = "mmseg" ]; then codebase_fullname="mmsegmentation"; fi
}


## parameters
export codebase=$1
export exec_performance=$2
export TENSORRT_VERSION=$3
export REQUIREMENT=$4 

getFullName $codebase
export MMDEPLOY_DIR=/root/workspace/mmdeploy
export REQ_DIR=${MMDEPLOY_DIR}/tests/jenkins/conf/${REQUIREMENT}

cp -R /root/workspace/jenkins/ mmdeploy/tests/
echo "start_time-$(date +%Y%m%d%H%M)"
## clone ${codebase}

branch=$(cat ${REQ_DIR} | xargs | sed 's/\s//g' | awk -F ${codebase}: '{print $2}' | awk -F '}' '{print $1}' | sed 's/,/\n/g' | grep branch | awk -F ':' '{print $2}')
git clone --branch ${branch} --depth 1 https://github.com/open-mmlab/${codebase_fullname}.git

## init tensorrt
if [[ "$TENSORRT_VERSION" = '8.4.1.5' ]]; then
    TENSORRT_DIR=/root/workspace/TensorRT-${TENSORRT_VERSION}
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH/8.2.5.1/${TENSORRT_VERSION}}
    cp -r cudnn-8.4.1.50/include/cudnn* /usr/local/cuda-11.3/include/
    cp -r cudnn-8.4.1.50/lib/libcudnn* /usr/local/cuda-11.3/lib64/ 
fi

## build mmdeploy
ln -s /root/workspace/mmdeploy_benchmark $MMDEPLOY_DIR/data

for TORCH_VERSION in 1.11.0; do
    conda activate torch${TORCH_VERSION}
    if [[ "$TENSORRT_VERSION" = '8.4.1.5' ]]; then
        pip install /root/workspace/TensorRT-8.4.1.5/python/tensorrt-8.4.1.5-cp38-none-linux_x86_64.whl
    fi
    # export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
    export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
    # need to build for each env
    mkdir -p $MMDEPLOY_DIR/build && cd $MMDEPLOY_DIR/build
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
        -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn" \
        -DMMDEPLOY_SHARED_LIBS=OFF \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DCUDNN_DIR=${CUDNN_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -Dncnn_DIR=${ncnn_DIR} \
        -DTorch_DIR=${Torch_DIR} \
        -Dpplcv_DIR=${pplcv_DIR} \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu"
    make -j $(nproc)
    make install && cd $MMDEPLOY_DIR

    pip install openmim
    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt
    pip install -r requirements/build.txt
    pip install -v .

    ## install requirements from conf
    mim install $(cat ${REQ_DIR} | xargs | sed 's/\s//g' | awk -F ${codebase}: '{print $2}' | awk -F '}' '{print $1}' | sed 's/,/\n/g' | grep -v branch | awk -F ':' '{print $2}')

    ## start regression
    log_dir=/root/workspace/mmdeploy_regression_working_dir/${codebase}/torch${TORCH_VERSION}
    log_path=${log_dir}/convert.log
    mkdir -p ${log_dir}
    # ignore pplnn as it's too slow
    python ./tools/regression_test.py \
        --codebase ${codebase} \
        --work-dir ${log_dir} \
        --device cuda:0 \
        --backends onnxruntime tensorrt ncnn torchscript openvino \
        ${exec_performance} 2>&1 | tee ${log_path}
done

echo "end_time-$(date +%Y%m%d%H%M)"
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.
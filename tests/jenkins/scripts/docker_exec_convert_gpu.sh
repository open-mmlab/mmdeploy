#!/bin/bash

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
getFullName $codebase
if [ '$2' -eq 'y' ]; then
    exec_performance = '--performance'
else
    exec_performance = ''
fi
export MMDEPLOY_DIR=/root/workspace/mmdeploy

echo "start_time-$(date +%Y%m%d%H%M)"
## clone ${codebase}
git clone --depth 1 https://github.com/open-mmlab/${codebase_fullname}.git /root/workspace/${codebase_fullname}

## build mmdeploy
ln -s /root/workspace/mmdeploy_benchmark $MMDEPLOY_DIR/data

for TORCH_VERSION in 1.10.0 1.11.0; do
    conda activate torch${TORCH_VERSION}
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

    ## build ${codebase}
    if [ ${codebase} == mmdet3d ]; then
        mim install ${codebase}
        mim install mmcv-full==1.5.2
        pip install -v /root/workspace/${codebase_fullname}
    elif [ ${codebase} == mmedit ]; then
        mim install ${codebase}
        mim install mmcv-full==1.6.0
        pip install -v /root/workspace/${codebase_fullname}
    elif [ ${codebase} == mmrotate ]; then
        mim install ${codebase}
        mim install mmcv-full==1.6.0
        pip install -v /root/workspace/${codebase_fullname}
    else
        mim install ${codebase}
        if [ $? -ne 0 ]; then
            mim install mmcv-full
            pip install -v /root/workspace/${codebase_fullname}
        fi
    fi
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

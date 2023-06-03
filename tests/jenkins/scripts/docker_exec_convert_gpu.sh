#!/bin/bash

set -e

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
    if [ "$codebase_" = "mmaction" ]; then codebase_fullname="mmaction2"; fi
    if [ "$codebase_" = "mmyolo" ]; then codebase_fullname="mmyolo"; fi
}

## prepare for mmdeploy test
export MMDEPLOY_DIR=/root/workspace/mmdeploy
export REGRESSION_DIR=/root/workspace/mmdeploy_regression_working_dir
ln -sf /root/workspace/mmdeploy_benchmark $MMDEPLOY_DIR/data
ln -sf /root/workspace/jenkins ${MMDEPLOY_DIR}/tests/jenkins

# install tensorrt
export TENSORRT_DIR=/root/workspace/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

export URL_PREFIX=$(cat ${REGRESSION_DIR}/host.cfg)
export HOST_LOG_PATH=$(cat ${REGRESSION_DIR}/log_path.cfg)

export codebase=$1
getFullName $codebase
export CONFIG=${MMDEPLOY_DIR}/tests/jenkins/conf/$2



## parameters
export exec_performance=$(grep exec_performance ${CONFIG} | sed 's/exec_performance=//')
export EXEC_MODELS=$(grep exec_models ${CONFIG} | sed 's/exec_models=//')
export EXEC_BACKENDS=$(grep exec_backends ${CONFIG} | sed 's/exec_backends=//')
export EXEC_TORCH_VERSIONS=$(grep exec_torch_versions ${CONFIG} | sed 's/exec_torch_versions=//')
export REQUIRE_JSON=${MMDEPLOY_DIR}/tests/jenkins/conf/$(grep requirement ${CONFIG} | sed 's/requirement=//')

if [[ "${exec_performance}" == "y" ]]; then
    export exec_performance="-p"
else
    export exec_performance=""
fi


echo "start_time-$(date +%Y%m%d%H%M)"
## clone ${codebase}

branch=$(cat ${REQUIRE_JSON} | xargs | sed 's/\s//g' | awk -F ${codebase}: '{print $2}' | awk -F '}' '{print $1}' | sed 's/,/\n/g' | grep branch | awk -F ':' '{print $2}')
git clone --branch ${branch} --depth 1 https://github.com/open-mmlab/${codebase_fullname}.git
export CODEBASE_ROOT_DIR=/root/workspace/${codebase_fullname}
# deal with mmyolo special
if [ ${codebase} == "mmyolo" ]; then
  ln -sf ${CODEBASE_ROOT_DIR}/configs/deploy ${MMDEPLOY_DIR}/configs/mmyolo
  ln -sf ${CODEBASE_ROOT_DIR}/tests/regression/mmyolo.yml ${MMDEPLOY_DIR}/tests/regression/mmyolo.yml
fi

for TORCH_VERSION in ${EXEC_TORCH_VERSIONS}; do
    conda activate torch${TORCH_VERSION}
    export PYTHON_VERSION=$(python -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}')
    pip install ${TENSORRT_DIR}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl
    # export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
    export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")

    if [ $TORCH_VERSION == "1.8.0" ]; then
        # fix torchscript issue of no libnvrtc-builtins.so.11.1
        export torch_lib_dir="$(python -m pip show torch | grep Location | awk '{print $2}')/torch"
        export target_file=${torch_lib_dir}/lib/libnvrtc-builtins.so.11.1
        if [ -f ${target_file} ]; then
            echo "File exits: ${target_file}"
        else
            cp -f ${torch_lib_dir}/lib/libnvrtc-builtins.so ${target_file}
        fi
    fi
    # need to build for each env
    mkdir -p $MMDEPLOY_DIR/build && cd $MMDEPLOY_DIR/build
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
        -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn;torchscript" \
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

    pip install openmim xlsxwriter clip
    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt
    pip install -r requirements/build.txt
    pip install -v .


    if [[ $codebase == "mmdet3d" ]] && [[ $branch == "dev-1.x" ]]; then
        echo "Install mmdet3d dev-1.x specially"
        mim install mmengine "mmcv>=2.0.0rc1"
        mim install /root/workspace/${codebase_fullname}
    elif [ "${codebase_fullname}" == "mmyolo"  ]; then
        pip install -e /root/workspace/mmyolo
        mim install 'mmdet>=3.0.0rc0'
        mim install 'mmcv>=2.0.0rc0' mmengine
    else
        ## install requirements from conf
        mim install $(cat ${REQUIRE_JSON} | xargs | sed 's/\s//g' | awk -F ${codebase}: '{print $2}' | awk -F '}' '{print $1}' | sed 's/,/\n/g' | grep -v branch | awk -F ':' '{print $2}')
    fi

    ## start regression
    log_dir=${REGRESSION_DIR}/${codebase}/torch${TORCH_VERSION}
    log_path=${log_dir}/convert_log.txt
    mkdir -p ${log_dir}
    # log env
    python tools/check_env.py 2>&1 | tee ${log_dir}/check_env_log.txt
    # ignore pplnn as it's too slow

    start_regression=$(date +%s)
    python ./tools/regression_test.py \
        --codebase ${codebase} \
        --work-dir ${log_dir} \
        --device cuda:0 \
        --models $EXEC_MODELS \
        --backends $EXEC_BACKENDS \
        ${exec_performance} 2>&1 | tee ${log_path}
    end_regression=$(date +%s)
    regression_time=$(( end_regression - start_regression ))
    echo "execution time regression of ${codebase} is ${regression_time} seconds"
    # get stats results
    python ${MMDEPLOY_DIR}/tests/jenkins/scripts/check_results.py \
        ${URL_PREFIX} \
        --host-log-path ${HOST_LOG_PATH} \
        --regression-dir ${REGRESSION_DIR}

done

echo "end_time-$(date +%Y%m%d%H%M)"
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.

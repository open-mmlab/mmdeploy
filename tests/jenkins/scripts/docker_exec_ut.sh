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

echo "start_time-$(date +%Y%m%d%H%M)"

## parameters
export MMDEPLOY_DIR=/root/workspace/mmdeploy
export UT_LOG_DIR=/root/workspace/ut_log

ln -sf /root/workspace/jenkins ${MMDEPLOY_DIR}/tests/jenkins

export CONFIG=${MMDEPLOY_DIR}/tests/jenkins/conf/$1
echo "Using config $CONFIG"
export TENSORRT_DIR=/root/workspace/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

## parameters
export EXEC_TORCH_VERSIONS=$(grep exec_torch_versions ${CONFIG} | sed 's/exec_torch_versions=//')
export REPO_VERSION=$(grep repo_version ${CONFIG} | sed 's/repo_version=//')

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
            cp ${torch_lib_dir}/lib/libnvrtc-builtins.so ${target_file}
        fi
    fi

    # need to build for each env
    # TODO add openvino
    mkdir -p $MMDEPLOY_DIR/build && cd $MMDEPLOY_DIR/build
    cmake .. -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
        -DMMDEPLOY_COVERAGE=ON \
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

    make -j $(nproc) && make install

    # sdk tests
    mkdir -p mmdeploy_test_resources/transform
    cp -f ../tests/data/tiger.jpeg mmdeploy_test_resources/transform/
    ./bin/mmdeploy_tests
    lcov --capture --directory . --output-file coverage.info
    ls -lah coverage.info
    cp -f coverage.info ${UT_LOG_DIR}/torch${TORCH_VERSION}_sdk_ut_converage.info

    cd $MMDEPLOY_DIR
    export mmcv_full="mmcv>=2.0.0rc0"
    if [ $REPO_VERSION == "v1" ]; then
        mmcv_full="mmcv-full==1.4.2"
    fi
    pip install -U openmim clip
    # fix E   AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)
    pip install opencv-python==4.5.4.60 opencv-python-headless==4.5.4.60 opencv-contrib-python==4.5.4.60
    mim install ${mmcv_full}
    pip install -r requirements/codebases.txt
    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt
    pip install -r requirements/build.txt
    pip install -v .

    ## start python tests
    set +e # ignore ut error and do not exit
    coverage run --branch --source mmdeploy -m pytest -rsE tests
    set -e # enable step error check
    coverage xml
    coverage report -m
    cp -f coverage.xml ${UT_LOG_DIR}/torch${TORCH_VERSION}_converter_converage.xml

done

echo "end_time-$(date +%Y%m%d%H%M)"
echo "end_time-$(date +%Y%m%d%H%M)"
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.

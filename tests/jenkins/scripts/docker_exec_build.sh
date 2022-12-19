#!/bin/bash

set -e
start=$(date +%s)
echo "start_time-$(date +%Y%m%d%H%M)"

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

export repo_version=${1:-v1.0}
export MMDEPLOY_DIR=/root/workspace/mmdeploy
export BUILD_LOG_DIR=/root/workspace/build_log
ln -s /root/workspace/mmdeploy_benchmark ${MMDEPLOY_DIR}/data
cp -R /root/workspace/jenkins ${MMDEPLOY_DIR}/tests/
export TENSORRT_DIR=/root/workspace/TensorRT
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

cd /root/workspace
mmdet_version=mmdet
mmdet_branch=master
if [ $repo_version == "v2.0" ]; then
    mmdet_version="mmdet>=3.0.0rc1"
    mmdet_branch=3.x
fi

git clone --depth 1 --branch $mmdet_branch https://github.com/open-mmlab/mmdetection.git

cd ${MMDEPLOY_DIR}
conda activate torch1.10.0
# export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
cd ${MMDEPLOY_DIR}
# install trt
export PYTHON_VERSION=$(python -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}')
pip install ${TENSORRT_DIR}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl

pip install -U openmim
mim install ${mmdet_version}
pip install -r requirements/tests.txt
pip install -r requirements/runtime.txt
pip install -r requirements/build.txt

## 校验动态库构建
mkdir -p ${MMDEPLOY_DIR}/build_share_libs && cd ${MMDEPLOY_DIR}/build_share_libs

cmake .. -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON -DMMDEPLOY_BUILD_TEST=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_ZIP_MODEL=ON \
    -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn;torchscript" \
    -DMMDEPLOY_SHARED_LIBS=ON \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DCUDNN_DIR=${CUDNN_DIR} \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -Dncnn_DIR=${ncnn_DIR} \
    -DTorch_DIR=${Torch_DIR} \
    -Dpplcv_DIR=${pplcv_DIR} \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu"

make -j $(nproc) && make install

cd ${MMDEPLOY_DIR}

pip install -v .
if [ $? -ne 0 ]; then
    echo "build mmdeploy with shared lib fail!"
    exit
fi

## 校验静态库构建
mkdir -p ${MMDEPLOY_DIR}/build && cd ${MMDEPLOY_DIR}/build
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
make -j $(nproc) && make install && cd ${MMDEPLOY_DIR}

pip install -v .
if [ $? -ne 0 ]; then
    echo "build mmdeploy with ubshared lib in torch1.8.0 fail!"
    exit
fi
test_log_dir=$BUILD_LOG_DIR/test_build
mkdir -p ${test_log_dir}
python tools/check_env.py 2>&1 | tee ${BUILD_LOG_DIR}/check_env_log.txt
python tools/regression_test.py --codebase mmdet --models ssd yolov3 --backends tensorrt onnxruntime \
    --performance --device cuda:0 --work-dir \
    ${test_log_dir} 2>&1 | tee ${test_log_dir}/mmdet_regresion_test_log.txt

## 校验不同torch版本下安装
for TORCH_VERSION in 1.10.0 1.11.0 1.12.0; do
    conda activate torch${TORCH_VERSION}
    pip install -v .
    if [ $? -ne 0 ]; then
        echo "build mmdeploy with ubshared lib in torch${TORCH_VERSION} fail!"
        exit
    fi
done
echo "end_time-$(date +%Y%m%d%H%M)"
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.

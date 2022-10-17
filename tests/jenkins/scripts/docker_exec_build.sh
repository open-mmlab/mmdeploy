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

export MMDEPLOY_DIR=/root/workspace/mmdeploy
echo "start_time-$(date +%Y%m%d%H%M)"
conda activate torch1.10.0
# export libtorch cmake dir, ran example: /opt/conda/envs/torch1.11.0/lib/python3.8/site-packages/torch/share/cmake/Torch
export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")
cd ${MMDEPLOY_DIR}

mim install mmcv-full
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
    -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn" \
    -DMMDEPLOY_SHARED_LIBS=ON \
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
    -DMMDEPLOY_TARGET_BACKENDS="trt;ort;ncnn" \
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

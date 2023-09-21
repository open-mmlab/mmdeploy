FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8

ARG MMDEPLOY_VERSION=main

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# build mmdeploy
RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy.git &&\
    export Torch_DIR=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')") &&\
    bash mmdeploy/.github/scripts/linux/build.sh "cpu;cuda" "ort;trt;ncnn;torchscript;openvino" \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -Dpplcv_DIR=${pplcv_DIR} \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DInferenceEngine_DIR=${InferenceEngine_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -Dncnn_DIR=${ncnn_DIR} \
        -DTorch_DIR=${Torch_DIR} &&\
    cd mmdeploy &&\
    python3 -m pip install -U openmim pycuda &&\
    python3 -m mim install "mmcv>=2.0.0" &&\
    python3 -m pip install -r requirements.txt &&\
    python3 -m pip install -e . --user

ENV MMDeploy_DIR="/root/workspace/mmdeploy/build/install/lib/cmake/MMDeploy"
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"
ENV PATH="/root/workspace/mmdeploy/build/bin:${PATH}"
ENV PYTHONPATH="/root/workspace/mmdeploy/build/lib:${PYTHONPATH}"

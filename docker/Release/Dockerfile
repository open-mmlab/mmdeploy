FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.3

ARG MMDEPLOY_VERSION

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# build mmdeploy
RUN git clone --recursive -b v$MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy.git &&\
    export Torch_DIR=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')") &&\
    mkdir -p mmdeploy/build && cd mmdeploy/build &&\
    cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -Dpplcv_DIR=${pplcv_DIR} \
      -DTENSORRT_DIR=${TENSORRT_DIR} \
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
      -DMMDEPLOY_TARGET_BACKENDS="ort;trt;ncnn;torchscript" \
      -Dncnn_DIR=${ncnn_DIR} \
      -DTorch_DIR=${Torch_DIR} \
      -DMMDEPLOY_CODEBASES=all &&\
    make -j$(nproc) && make install && cd .. &&\
    python3 -m pip install -U openmim pycuda &&\
    python3 -m pip install -r requirements.txt &&\
    python3 -m pip install -e .

ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/install/lib:${BACKUP_LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/mmdeploy/lib:${LD_LIBRARY_PATH}"
ENV PATH="/root/workspace/mmdeploy/install/bin:${PATH}"

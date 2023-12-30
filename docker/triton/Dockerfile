FROM nvcr.io/nvidia/tritonserver:22.12-pyt-python-py3

ARG CUDA=11.3
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"
ARG MMCV_VERSION=">=2.0.0rc2"
ARG MMENGINE_VERSION=">=0.3.0"

WORKDIR /root/workspace

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.sh &&\
    bash cmake-3.26.3-linux-x86_64.sh --skip-license --prefix=/usr

RUN git clone --depth 1 --branch v${PPLCV_VERSION} https://github.com/openppl-public/ppl.cv.git  &&\
    cd ppl.cv &&\
    ./build.sh cuda &&\
    mv cuda-build/install ./ &&\
    rm -rf cuda-build
ENV pplcv_DIR=/root/workspace/ppl.cv/install/lib/cmake/ppl

RUN apt-get update &&\
    apt-get install -y libopencv-dev

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=/root/workspace/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/lib:$LD_LIBRARY_PATH

RUN python3 -m pip install -U pip &&\
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html &&\
    pip install openmim &&\
    mim install "mmcv"${MMCV_VERSION} onnxruntime-gpu==${ONNXRUNTIME_VERSION} mmengine${MMENGINE_VERSION} &&\
    ln /usr/bin/python3 /usr/bin/python

COPY TensorRT-8.2.3.0 /root/workspace/tensorrt
RUN pip install /root/workspace/tensorrt/python/*cp38*whl
ENV TENSORRT_DIR=/root/workspace/tensorrt
ENV LD_LIBRARY_PATH=/root/workspace/tensorrt/lib:$LD_LIBRARY_PATH

RUN apt-get install -y rapidjson-dev

RUN git clone -b v1.0.0rc7 https://github.com/open-mmlab/mmpretrain.git &&\
    cd mmpretrain && pip install .

RUN git clone -b v3.0.0 https://github.com/open-mmlab/mmdetection.git &&\
    cd mmdetection && pip install .

RUN git clone -b v1.0.0 https://github.com/open-mmlab/mmsegmentation.git &&\
    cd mmsegmentation && pip install .

RUN git clone -b v1.0.0 https://github.com/open-mmlab/mmocr.git &&\
    cd mmocr && pip install .

RUN git clone -b v1.0.0rc1 https://github.com/open-mmlab/mmrotate.git &&\
    cd mmrotate && pip install .

RUN git clone -b v1.0.0 https://github.com/open-mmlab/mmpose.git &&\
    cd mmpose && pip install .

RUN git clone -b triton-server --recursive https://github.com/irexyc/mmdeploy &&\
    cd mmdeploy && mkdir -p build && cd build &&\
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON  \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_BUILD_TEST=OFF \
        -DMMDEPLOY_TARGET_BACKENDS="trt;ort" \
        -DMMDEPLOY_CODEBASES=all \
        -Dpplcv_DIR=${pplcv_DIR} \
        -DMMDEPLOY_BUILD_EXAMPLES=OFF \
        -DMMDEPLOY_DYNAMIC_BACKEND=OFF \
        -DTRITON_MMDEPLOY_BACKEND=ON \
        -DTRITON_TAG="r22.12" &&\
    make -j$(nproc) && make install &&\
    cp -r install/backends /opt/tritonserver/ &&\
    cd .. && pip install -e . --user

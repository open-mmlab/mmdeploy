ARG CUDA_INT=118

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS cuda-118
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 AS cuda-113

FROM cuda-${CUDA_INT} AS final

ARG TORCH_VERSION=2.0.0
ARG TORCHVISION_VERSION=0.15.1

# important dependencies
ARG OPENCV_VERSION==4.5.4.60
ARG PPLCV_VERSION=0.7.0

# backends
ARG ONNXRUNTIME_VERSION=1.15.1
ARG PPLNN_VERSION=0.8.1
ARG NCNN_VERSION=20230816
ARG TENSORRT_VERSION=8.6.1.6
ARG OPENVINO_VERSION=2022.3.0

# tensorrt tar file url
ARG TENSORRT_URL

USER root
WORKDIR /root/workspace

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        ca-certificates \
        gcc-7 \
        g++-7 \
        git \
        vim \
        wget \
        libopencv-dev \
        libprotobuf-dev protobuf-compiler \
        unzip \
        python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# install jdk, onnxruntime, openvino, and other python packages
RUN wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz &&\
    tar xvf openjdk-18_linux-x64_bin.tar.gz && rm -rf openjdk-18_linux-x64_bin.tar.gz && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -xzvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -xzvf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && rm onnxruntime-*.tgz &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.tar.gz &&\
    tar -xzvf cmake-3.25.2-linux-x86_64.tar.gz && rm cmake-*.tar.gz && mv cmake-* cmake &&\
    export CUDA_INT=$(echo $CUDA_VERSION | awk '{split($0, a, "."); print a[1]a[2]}') &&\
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel &&\
    python3 -m pip install --no-cache-dir onnxruntime-gpu==${ONNXRUNTIME_VERSION} &&\
    python3 -m pip install --no-cache-dir openvino openvino-dev[onnx]==${OPENVINO_VERSION} &&\
    python3 -m pip install --no-cache-dir opencv-python==${OPENCV_VERSION} opencv-python-headless==${OPENCV_VERSION} opencv-contrib-python==${OPENCV_VERSION} &&\
    python3 -m pip install --no-cache-dir torch==${TORCH_VERSION}+cu${CUDA_INT} torchvision==${TORCHVISION_VERSION}+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html

# create env
ENV JAVA_HOME=/root/workspace/jdk-18
ENV PATH=$JAVA_HOME/bin:/root/workspace/cmake/bin:$PATH
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_GPU_DIR=/root/workspace/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$LD_LIBRARY_PATH

### install ppl.nn
RUN git clone --depth 1 --branch v${PPLNN_VERSION} --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    export PYTHON_INCLUDE_DIR=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])') &&\
    ./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_PYTHON_API=ON -DPYTHON3_INCLUDE_DIRS=${PYTHON_INCLUDE_DIR} &&\
    cd ./python/package && \
    ./build.sh && \
    cd - && mv pplnn-build/install ./ && rm -rf pplnn-build &&\
    cd /tmp/pyppl-package/dist && \
    python3 -m pip install pyppl*.whl && rm *.whl

ENV pplnn_DIR=/root/workspace/ppl.nn/install/lib/cmake/ppl
ENV PYTHONPATH=/root/workspace/ppl.nn/install/lib:$PYTHONPATH


### build ncnn
RUN git clone --depth 1 --branch ${NCNN_VERSION} --recursive https://github.com/Tencent/ncnn.git &&\
    python3 -m pip install pybind11 &&\
    mkdir ncnn/build && cd ncnn/build &&\
    cmake -DNCNN_VULKAN=OFF -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=OFF -DCMAKE_INSTALL_PREFIX=../install .. &&\
    make -j $(nproc) && make install &&\
    cd .. && python3 -m pip install . && rm -rf ./build

ENV ncnn_DIR=/root/workspace/ncnn/install/lib/cmake/ncnn

### install ppl.cv
RUN git clone --depth 1 --branch v${PPLCV_VERSION} https://github.com/openppl-public/ppl.cv.git  &&\
    cd ppl.cv &&\
    ./build.sh cuda && mv cuda-build/install ./ && rm -rf cuda-build

ENV pplcv_DIR=/root/workspace/ppl.cv/install/lib/cmake/ppl

### install tensorrt
RUN wget -c $TENSORRT_URL && \
    tar -zxvf /root/workspace/TensorRT-${TENSORRT_VERSION}*.tar.gz -C /root/workspace &&\
    rm -rf TensorRT-${TENSORRT_VERSION}*.tar.gz &&\
    ln -sf /root/workspace/TensorRT-${TENSORRT_VERSION} /root/workspace/TensorRT &&\
    cd /root/workspace/TensorRT && rm -rf data doc samples uff &&\
    export PY_VERSION=$(python3 -V | awk '{print $2}' | awk '{split($0, a, "."); print a[1]a[2]}') &&\
    python3 -m pip install ./python/tensorrt-*-cp${PY_VERSION}-none-linux_x86_64.whl

ENV TENSORRT_DIR=/root/workspace/TensorRT
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
ENV PATH=$TENSORRT_DIR/bin:$PATH

# openvino
RUN wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/l_openvino_toolkit_ubuntu20_2022.3.0.9052.9752fafe8eb_x86_64.tgz &&\
    tar -zxvf ./l_openvino_toolkit*.tgz &&\
    rm ./l_openvino_toolkit*.tgz &&\
    mv ./l_openvino* ./openvino_toolkit &&\
    bash ./openvino_toolkit/install_dependencies/install_openvino_dependencies.sh

ENV OPENVINO_DIR=/root/workspace/openvino_toolkit
ENV InferenceEngine_DIR=$OPENVINO_DIR/runtime/cmake

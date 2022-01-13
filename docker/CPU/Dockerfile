ARG OS_VERSION=18.04
FROM ubuntu:${OS_VERSION} as official
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.8.0
ARG TORCHVISION_VERSION=0.9.0
ARG ONNXRUNTIME_VERSION=1.8.1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        libjpeg-dev \
        libpng-dev \
        ccache \
        cmake \
        gcc \
        g++ \
        git \
        vim \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### pytorch
RUN /opt/conda/bin/pip install torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
ENV PATH /opt/conda/bin:$PATH

### install open-mim
RUN /opt/conda/bin/pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VERSION}/index.html

WORKDIR /root/workspace
RUN git clone https://github.com/open-mmlab/mmclassification


### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### update cmake to 20
RUN apt-get update && apt-get install libssl-dev &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
    tar -zxvf cmake-3.20.0.tar.gz &&\
    cd cmake-3.20.0 &&\
    ./bootstrap &&\
    make &&\
    make install &&\
    rm -rf /var/lib/apt/lists/*

## install onnxruntme and openvino
RUN /opt/conda/bin/pip install onnxruntime==${ONNXRUNTIME_VERSION} openvino-dev

## build ncnn
RUN apt-get update && apt-get install libprotobuf-dev protobuf-compiler -y --no-install-recommends &&\
    git clone https://github.com/Tencent/ncnn.git &&\
    cd ncnn &&\
    export NCNN_DIR=$(pwd) &&\
    git submodule update --init &&\
    mkdir -p build && cd build &&\
    cmake -DNCNN_VULKAN=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON .. &&\
    make install &&\
    cd ${NCNN_DIR} # To NCNN root directory &&\
    cd python &&\
    pip install -e .

## build ppl.nn
WORKDIR /root/workspace
RUN git clone https://github.com/openppl-public/ppl.nn &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_ENABLE_PYTHON_API=ON -DHPCC_USE_X86_64=ON &&\
    rm -rf /tmp/pyppl-package &&\
    cp -r python/package /tmp/pyppl-package &&\
    cp -r pplnn-build/install/lib/pyppl/* /tmp/pyppl-package/pyppl &&\
    cd /tmp/pyppl-package &&\
    pip3 install .

### install mmdeploy
WORKDIR /root/workspace
RUN git clone https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    git submodule update --init --recursive &&\
    rm -rf build &&\
    mkdir build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=/root/workspace/ncnn/build/install/lib/cmake/ncnn .. &&\
    make -j$(nproc) &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS=ort .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    pip install -e .

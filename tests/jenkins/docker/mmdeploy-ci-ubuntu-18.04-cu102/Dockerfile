FROM nvcr.io/nvidia/tensorrt:20.03-py3
ARG PYTHON_VERSION=3.8
ARG CUDA=10.2
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLNN_VERSION=0.8.1
ARG TENSORRT=7.0.0
ARG CUDNN=7.6.5
ARG OPENCV_VERSION==4.5.5.62
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG PPLCV_VERSION=0.7.0
ARG NCNN_VERSION=20220721
USER root


WORKDIR /root/workspace
ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

### change the system source for installing libs
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        libssl-dev \
        libprotobuf-dev protobuf-compiler \
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
        libzip-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

### install miniconda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH

### install jdk1.8
RUN wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz &&\
tar xvf openjdk-18_linux-x64_bin.tar.gz && rm -rf openjdk-18_linux-x64_bin.tar.gz
ENV JAVA_HOME=/root/workspace/jdk-18
ENV PATH=$JAVA_HOME/bin:$PATH

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### install opencv
RUN /opt/conda/bin/pip install opencv-python==${OPENCV_VERSION}

### install openmim
RUN /opt/conda/bin/pip install openmim

### install ppl.nn
RUN git clone --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    git checkout tags/v${PPLNN_VERSION} -b ${PPLNN_VERSION}

ENV pplnn_DIR=/root/workspace/ppl.nn/install/lib/cmake/ppl
ENV PYTHONPATH=/root/workspace/ppl.nn/install/lib:$PYTHONPATH

### build ncnn
RUN git clone --recursive https://github.com/Tencent/ncnn.git &&\
    cd ncnn &&\
    git checkout tags/${NCNN_VERSION} -b ${NCNN_VERSION} && \
    export NCNN_DIR=$(pwd) &&\
    mkdir -p build && cd build &&\
    cmake -DNCNN_VULKAN=OFF -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DCMAKE_INSTALL_PREFIX=$NCNN_DIR/install .. &&\
    make -j $(nproc) && make install &&\
    cd $NCNN_DIR/python &&\
    /opt/conda/bin/pip install -e . 

### build sdk
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda
ENV pplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl


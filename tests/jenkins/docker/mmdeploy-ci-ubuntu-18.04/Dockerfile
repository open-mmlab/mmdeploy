FROM openvino/ubuntu18_dev:2021.4.2
ARG PYTHON_VERSION=3.7
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG MMCV_VERSION=1.5.3
USER root

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
        libopencv-dev libspdlog-dev \
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
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### install open-mim
RUN /opt/conda/bin/pip install openmim

WORKDIR /root/workspace

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### install onnxruntme and openvino
RUN /opt/conda/bin/pip install onnxruntime==${ONNXRUNTIME_VERSION} openvino-dev

### build ncnn
RUN git clone https://github.com/Tencent/ncnn.git &&\
    cd ncnn &&\
    export NCNN_DIR=$(pwd) &&\
    git submodule update --init &&\
    mkdir -p build && cd build &&\
    cmake -DNCNN_VULKAN=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON .. &&\
    make -j$(nproc) &&\
    make install &&\
    cd /root/workspace/ncnn/python &&\
    pip install -e .

### create conda env && install pytorch
RUN /opt/conda/bin/conda init bash && \
    TORCH_VERSION_LIST=(1.8.0 1.9.0 1.10.0 1.11.0 1.12.0) && \
    TORCHVISION_VERSION_LIST=(0.9.0 1.10.0 0.11.0 0.12.0 0.13.0) && \
    for i in `seq 0 4`; \
    do \
        /opt/conda/bin/conda create --name torch${TORCH_VERSION_LIST[$i]} python=3.8 -y; \
        /opt/conda/bin/conda run -n torch${TORCH_VERSION_LIST[$i]} /opt/conda/bin/pip install torch==${TORCH_VERSION_LIST[$i]}+cpu torchvision==${TORCHVISION_VERSION_LIST[$i]}+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html; \
    done && \
    PYTHON_VERSION=(3.6 3.7 3.8 3.9) && \
    for i in `seq 0 3`; \
    do \
        /opt/conda/bin/conda create --name mmdeploy-${PYTHON_VERSION[i]} python=${PYTHON_VERSION[i]} -y; \
    done

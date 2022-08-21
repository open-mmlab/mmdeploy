FROM nvcr.io/nvidia/tensorrt:22.03-py3

ARG PYTHON_VERSION=3.8
ARG CUDA=11.3
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLNN_VERSION=0.8.1
ARG CUDNN=8.2.1
ARG OPENCV_VERSION==4.5.5.62
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG MMCV_VERSION=1.5.3
ARG PPLNN_version=0.8.1
ARG PPLCV_VERSION=0.7.0
USER root


WORKDIR /root/workspace
ENV DEBIAN_FRONTEND=nointeractive


### change the system source for installing libs
ARG USE_SRC_INSIDE=true
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

### update apt and install libs
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

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### install jdk1.8
RUN wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz &&\
tar xvf openjdk-18_linux-x64_bin.tar.gz && rm -rf openjdk-18_linux-x64_bin.tar.gz
ENV JAVA_HOME=/root/workspace/jdk-18
ENV PATH=$JAVA_HOME/bin:$PATH

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch
ENV PATH /opt/conda/bin:$PATH


WORKDIR /root/workspace

### install openmim
RUN /opt/conda/bin/pip install openmim

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### install onnxruntme and openvino
RUN /opt/conda/bin/pip install onnxruntime==${ONNXRUNTIME_VERSION} openvino-dev

### install opencv
RUN /opt/conda/bin/pip install opencv-python==${OPENCV_VERSION}

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
ENV ncnn_DIR=/root/workspace/ncnn/install/lib/cmake/ncnn
ENV PYTHONPATH=/root/workspace/ncnn/python:$PYTHONPATH 

### install ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

### install libtorch
RUN wget https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip && \
    unzip libtorch-shared-with-deps-1.8.1+cu111.zip && \
    cd libtorch
ENV Torch_DIR=/root/workspace/libtorch
ENV LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH

### cp trt from pip to conda
RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/
ENV TENSORRT_DIR=/workspace/tensorrt

### create conda env && install pytorch
RUN /opt/conda/bin/conda init bash && \
    TORCH_VERSION_LIST=(1.8.0 1.9.0 1.10.0 1.11.0 1.12.0) && \
    TORCHVISION_VERSION_LIST=(0.9.0 0.10.0 0.11.0 0.12.0 0.13.0) && \
    for i in `seq 0 1`; \
    do \
        /opt/conda/bin/conda create --name torch${TORCH_VERSION_LIST[$i]} --clone base -y; \
        /opt/conda/bin/conda run -n torch${TORCH_VERSION_LIST[$i]} /opt/conda/envs/torch${TORCH_VERSION_LIST[$i]}/bin/pip install torch==${TORCH_VERSION_LIST[$i]}+cpu torchvision==${TORCHVISION_VERSION_LIST[$i]}+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html; \
    done 
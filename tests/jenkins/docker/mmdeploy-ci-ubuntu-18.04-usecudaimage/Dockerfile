FROM nvidia/cuda:10.2-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7
ARG CUDA=10.2
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLNN_VERSION=0.8.1
ARG TENSORRT=7.0.0
ARG CUDNN=7.6
ARG OPENCV_VERSION==4.5.5.62
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG PPLCV_VERSION=0.7.0
ARG NCNN_VERSION=20220721
ARG HOST
USER root


WORKDIR /root/workspace
ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

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

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        libssl-dev \
        libprotobuf-dev protobuf-compiler \
        build-essential \
        libjpeg-dev \
        libpng-dev \
        ccache \
        gcc \
        g++ \
        git \
        vim \
        wget \
        curl \
        libzip-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

### install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1.tar.gz && \
    tar -zxvf cmake-3.24.1.tar.gz && \
    cd cmake-3.24.1 && \
    ./bootstrap && \
    make && \
    make install 

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

### install tensorrt
RUN wget http://${HOST}/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz && \
    tar -zxvf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz  &&\
    /opt/conda/bin/pip install TensorRT-7.0.0.11/python/tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl &&\
    /opt/conda/bin/pip install pycuda
ENV TENSORRT_DIR=/root/workspace/TensorRT-7.0.0.11
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

### install cudnn
RUN wget http://${HOST}/cudnn-10.2-linux-x64-v7.6.5.32.tgz && \
    tar -zxvf cudnn-10.2-linux-x64-v7.6.5.32.tgz && \
    cp -r cuda/include/cudnn.h* /usr/local/cuda-10.2/include/ && \
    cp -r cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64/
ENV CUDNN_DIR=/root/workspace/cuda
ENV LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    /opt/conda/bin/pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### install openvino
RUN /opt/conda/bin/pip install openvino-dev

### install opencv
RUN /opt/conda/bin/pip install opencv-python==${OPENCV_VERSION}

### install ppl.nn
RUN git clone --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_CUDA=ON
ENV PPLNN_DIR=/root/workspace/ppl.nn 
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

### install zip
RUN apt update && apt install -y zip 
### install libtorch
RUN wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.12.1%2Bcu102.zip && \
    unzip libtorch-shared-with-deps-1.12.1+cu102.zip 
ENV Torch_DIR=/root/workspace/libtorch
ENV LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH

### install openmim
RUN /opt/conda/bin/pip install openmim


### create conda env && install pytorch
RUN /opt/conda/bin/conda init bash && \
    TORCH_VERSION_LIST=(1.9.0 1.10.0 1.11.0 1.12.0) && \
    TORCHVISION_VERSION_LIST=(0.10.0 0.11.0 0.12.0 0.13.0) && \
    for i in `seq 0 1`; \
    do \
        /opt/conda/bin/conda create --name torch${TORCH_VERSION_LIST[$i]} --clone base -y; \
        /opt/conda/bin/conda install -n torch${TORCH_VERSION_LIST[$i]} -y pytorch==${TORCH_VERSION_LIST[i]} torchvision==${TORCHVISION_VERSION_LIST[i]} cudatoolkit=${CUDA} -c pytorch -c conda-forge; \
    done 
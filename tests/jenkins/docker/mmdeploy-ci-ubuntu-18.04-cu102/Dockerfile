FROM nvidia/cuda:10.2-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7
ARG CUDA=10.2
ARG CUDA_INT=102
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
ENV CUDA_VERSION=10.2
ENV PYTHON_VERSION=3.7
### change the system source for installing libs
ARG USE_SRC_INSIDE=true
RUN if [ ${USE_SRC_INSIDE} ] ; \
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
        gcc-7 \
        g++-7 \
        git \
        vim \
        wget \
        curl \
        libzip-dev \
        libopencv-dev \
        libpython${PYTHON_VERSION} \
        lcov \
        zip \
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
RUN wget http://${HOST}/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz && \
    tar -zxvf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz  &&\
    /opt/conda/bin/pip install TensorRT-7.2.3.4/python/tensorrt-7.2.3.4-cp37-none-linux_x86_64.whl &&\
    /opt/conda/bin/pip install pycuda
ENV TENSORRT_DIR=/root/workspace/TensorRT-7.2.3.4
ENV TENSORRT_VERSION=7.2.3.4
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

### install cudnn
RUN wget http://${HOST}/cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
    tar -zxvf cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
    cp -r cuda/include/cudnn* /usr/local/cuda-${CUDA}/include/ && \
    cp -r cuda/lib64/libcudnn* /usr/local/cuda-${CUDA}/lib64/
ENV CUDNN_DIR=/root/workspace/cuda
ENV LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64/:$LD_LIBRARY_PATH
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA}

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    /opt/conda/bin/pip install onnxruntime==${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

### install openvino
RUN /opt/conda/bin/pip install openvino-dev

### install opencv
RUN /opt/conda/bin/pip install opencv-python==${OPENCV_VERSION}

### install ppl.nn
RUN git clone --depth 1 --branch v${PPLNN_VERSION} --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_PYTHON_API=ON -DPYTHON3_INCLUDE_DIRS=/opt/conda/include/python3.7 && \
    cd ./python/package && \
    ./build.sh && \
    cd /tmp/pyppl-package/dist && \
    /opt/conda/bin/pip install pyppl*.whl
ENV PPLNN_DIR=/root/workspace/ppl.nn
ENV pplnn_DIR=${PPLNN_DIR}/pplnn-build/install/lib/cmake/ppl
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
ENV pplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl

### create conda env && install pytorch
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda create --clone base -n torch1.8.1 -y && \
    /opt/conda/envs/torch1.8.1/bin/pip install torch==1.8.1+cu${CUDA_INT} torchvision==0.9.1+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda create --clone base -n torch1.9.0 -y && \
    /opt/conda/envs/torch1.9.0/bin/pip install torch==1.9.0+cu${CUDA_INT} torchvision==0.10.0+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda create --clone base -n torch1.10.0 -y && \
    /opt/conda/envs/torch1.10.0/bin/pip install torch==1.10.0+cu${CUDA_INT} torchvision==0.11.0+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda create --clone base -n torch1.11.0 -y && \
    /opt/conda/envs/torch1.11.0/bin/pip install torch==1.11.0+cu${CUDA_INT} torchvision==0.12.0+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda create --clone base -n torch1.12.0 -y && \
    /opt/conda/envs/torch1.12.0/bin/pip install torch==1.12.0+cu${CUDA_INT} torchvision==0.13.0+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html

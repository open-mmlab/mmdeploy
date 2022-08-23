FROM nvcr.io/nvidia/tensorrt:22.04-py3

ARG PYTHON_VERSION=3.8
ARG CUDA=11.3
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLNN_VERSION=0.8.1
ARG CUDNN=8.2.1
ARG OPENCV_VERSION==4.5.5.62
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ARG NCNN_VERSION=20220721
ENV FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

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

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

### install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz && \
    ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/

### install jdk1.8
RUN wget https://download.java.net/java/GA/jdk18/43f95e8614114aeaa8e8a5fcf20a682d/36/GPL/openjdk-18_linux-x64_bin.tar.gz &&\
tar xvf openjdk-18_linux-x64_bin.tar.gz && rm -rf openjdk-18_linux-x64_bin.tar.gz
ENV JAVA_HOME=/root/workspace/jdk-18
ENV PATH=$JAVA_HOME/bin:$PATH

### install openmim
RUN /opt/conda/bin/pip install openmim

WORKDIR /root/workspace
### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

### cp trt from pip to conda
RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/
ENV TENSORRT_DIR=/workspace/tensorrt

### install openvino
RUN /opt/conda/bin/pip install openvino-dev
RUN git clone --recursive https://github.com/openvinotoolkit/openvino.git && \
    cd openvino && \
    chmod +x install_build_dependencies.sh && \
    ./install_build_dependencies.sh && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make --jobs=$(nproc --all)

### install opencv
RUN /opt/conda/bin/pip install opencv-python==${OPENCV_VERSION}

### install ppl.nn
RUN git clone --recursive https://github.com/openppl-public/ppl.nn.git &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_USE_X86_64=ON  
ENV pplnn_DIR=/root/workspace/ppl.nn 
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
ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

### install zip
RUN apt update && apt install -y zip 
### install libtorch
RUN wget https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.1%2Bcu113.zip && \
    unzip libtorch-shared-with-deps-1.12.1+cu113.zip 
ENV Torch_DIR=/root/workspace/libtorch
ENV LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH


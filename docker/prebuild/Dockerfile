FROM quay.io/pypa/manylinux2014_x86_64

# package urls
ARG CUDA_URL
ARG CUDNN_URL
ARG TENSORRT_URL

ARG CUDA_VERSION=11.8

# important dependencies
ARG OPENCV_VERSION=4.5.5
ARG PPLCV_VERSION=0.7.0

# backends
ARG ONNXRUNTIME_VERSION=1.15.1
ARG TENSORRT_VERSION=8.6.1.6

# torch
ARG TORCH_VERSION=2.0.0
ARG TORCHVISION_VERSION=0.15.0

ARG TOOLSET_VERSION=7

USER root
WORKDIR /root/workspace

ENV FORCE_CUDA="1"

# install toolset
RUN yum install centos-release-scl devtoolset-${TOOLSET_VERSION}-gcc* -y

ENV TOOLSET_DIR=/opt/rh/devtoolset-${TOOLSET_VERSION}/root/usr
ENV PATH=$TOOLSET_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$TOOLSET_DIR/lib:$TOOLSET_DIR/lib64:/usr/local/lib64

# install cuda cudnn
RUN curl -fsSL -v -o ./cuda_install.run -O $CUDA_URL &&\
    chmod +x ./cuda_install.run &&\
    ./cuda_install.run --silent --toolkit &&\
    rm -f ./cuda_install.run &&\
    curl -fsSL -v -o ./cudnn.tgz -O $CUDNN_URL &&\
    tar -xzvf ./cudnn.tgz &&\
    rm -f ./cudnn.tgz &&\
    mv cu* /opt/cudnn

# install ort, trt
RUN curl -fsSL -v -o ./onnxruntime.tgz -O https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -xzvf onnxruntime.tgz &&\
    rm onnxruntime.tgz &&\
    mv onnxruntime* /opt/onnxruntime &&\
    curl -fsSL -v -o ./onnxruntime.tgz -O https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -xzvf onnxruntime.tgz &&\
    rm onnxruntime.tgz &&\
    mv onnxruntime* /opt/onnxruntime-gpu &&\
    curl -fsSL -v -o ./tensorrt.tgz -O $TENSORRT_URL &&\
    tar -xzvf ./tensorrt.tgz &&\
    rm -f ./tensorrt.tgz &&\
    mv ./TensorRT* /opt/TensorRT &&\
    cd /opt/TensorRT &&\
    rm -rf data doc samples uff

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDNN_DIR=/opt/cudnn
ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV ONNXRUNTIME_GPU_DIR=/opt/onnxruntime-gpu
ENV TENSORRT_DIR=/opt/TensorRT

ENV LD_LIBRARY_PATH=$CUDA_TOOLKIT_ROOT_DIR/lib64:$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_GPU_DIR}/lib:$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
ENV PATH=$TENSORRT_DIR/bin:$PATH

### install ppl.cv
RUN git clone --depth 1 --branch v${PPLCV_VERSION} https://github.com/openppl-public/ppl.cv.git  &&\
    cd ppl.cv &&\
    ./build.sh cuda &&\
    mv cuda-build/install ./ &&\
    rm -rf cuda-build

ENV pplcv_DIR=/root/workspace/ppl.cv/install/lib/cmake/ppl

# build opencv as static lib
RUN curl -fsSL -v -o ./opencv.tgz -O https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz &&\
    tar -xzvf ./opencv.tgz &&\
    rm -f ./opencv.tgz &&\
    cd opencv-${OPENCV_VERSION} &&\
    mkdir build && cd build &&\
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_SHARED_LIBS=OFF &&\
    make -j$(nproc) && make install

ENV OpenCV_DIR=/usr/local/lib64/cmake/opencv4

# install conda env
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    bash ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda create -n mmdeploy-3.6 python=3.6 -y &&\
    /opt/conda/bin/conda create -n mmdeploy-3.7 python=3.7 -y &&\
    /opt/conda/bin/conda create -n mmdeploy-3.8 python=3.8 -y &&\
    /opt/conda/bin/conda create -n mmdeploy-3.9 python=3.9 -y &&\
    /opt/conda/bin/conda create -n mmdeploy-3.10 python=3.10 -y &&\
    /opt/conda/bin/conda create -n mmdeploy-3.11 python=3.11 -y &&\
    export CUDA_INT=$(echo $CUDA_VERSION | awk '{split($0, a, "."); print a[1]a[2]}') &&\
    /opt/conda/bin/conda create -n torch${TORCH_VERSION} python=3.8 -y &&\
    /opt/conda/envs/mmdeploy-3.6/bin/pip install --no-cache-dir setuptools wheel pyyaml packaging &&\
    /opt/conda/envs/torch${TORCH_VERSION}/bin/pip install --no-cache-dir onnxruntime-gpu==${ONNXRUNTIME_VERSION} &&\
    /opt/conda/envs/torch${TORCH_VERSION}/bin/pip install ${TENSORRT_DIR}/python/tensorrt-*-cp38-none-linux_x86_64.whl &&\
    /opt/conda/envs/torch${TORCH_VERSION}/bin/pip install --no-cache-dir torch==${TORCH_VERSION}+cu${CUDA_INT} \
        torchvision==${TORCHVISION_VERSION}+cu${CUDA_INT} -f https://download.pytorch.org/whl/torch_stable.html &&\
    /opt/conda/bin/conda init bash &&\
    /opt/conda/bin/conda clean -ya

ENV CONDA=/opt/conda
ENV PATH=$CONDA/bin:$PATH

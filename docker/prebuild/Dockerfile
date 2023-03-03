FROM quay.io/pypa/manylinux2014_x86_64

# package urls
ARG CUDA_URL
ARG CUDNN_URL
ARG TENSORRT_URL


# important dependencies
ARG OPENCV_VERSION=4.5.5
ARG PPLCV_VERSION=0.7.0

# backends
ARG ONNXRUNTIME_VERSION=1.8.1

# torch
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0

USER root
WORKDIR /root/workspace

ENV FORCE_CUDA="1"

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
    curl -fsSL -v -o ./tensorrt.tgz -O $TENSORRT_URL &&\
    tar -xzvf ./tensorrt.tgz &&\
    rm -f ./tensorrt.tgz &&\
    mv ./TensorRT* /opt/TensorRT &&\
    cd /opt/TensorRT &&\
    rm -rf data doc samples uff

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV CUDNN_DIR=/opt/cudnn
ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV TENSORRT_DIR=/opt/TensorRT

ENV LD_LIBRARY_PATH=$CUDA_TOOLKIT_ROOT_DIR/lib64:$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
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

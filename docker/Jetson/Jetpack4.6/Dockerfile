FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

ARG MMDEPLOY_VERSION=main
ENV NVIDIA_VISIBLE_DEVICE all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.6/site-packages/opencv-python.libs/${LD_LIBRARY_PATH}"
ENV TENSORRT_DIR="/usr/include/aarch64-linux-gnu"

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

USER root
WORKDIR /root/workspace

# install dependencies
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA &&\
     apt-get update &&\ 
     apt-get install -y vim wget libspdlog-dev libssl-dev libpng-dev pkg-config libhdf5-100 libhdf5-dev --no-install-recommends

RUN python3 -m pip install --upgrade pip &&\    
    python3 -m pip install onnx==1.10 versioned-hdf5

# install onnxruntime
RUN wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kcnn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl &&\
    python3 -m pip install --no-cache-dir onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

# install mmcv
RUN git clone --branch 2.x https://github.com/open-mmlab/mmcv.git
RUN cd mmcv &&\ 
    python3 -m pip install --no-cache-dir opencv-python==4.5.4.60 &&\
    MMCV_WITH_OPS=1 python3 -m pip install -e .

# build ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    echo "export PPLCV_DIR=/root/workspace/ppl.cv" >> ~/.bashrc &&\
    cd ppl.cv &&\
    ./build.sh cuda
  
# download mmdeploy
RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy

# build TRT custom op
RUN cd mmdeploy &&\
    mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR    &&\
    make -j$(nproc) && make install
RUN cd mmdeploy &&\ 
    python3 -m pip install --upgrade setuptools &&\ 
    python3 -m pip install -e .

# build mmdeploy
RUN cd mmdeploy &&\
    mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_BUILD_SDK=ON \
    	-DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    	-DMMDEPLOY_BUILD_EXAMPLES=ON \
    	-DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR \ 
    	-Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
    	-DMMDEPLOY_CODEBASES=all && \
    make -j$(nproc) && make install

ENV MMDeploy_DIR="/root/workspace/mmdeploy/build/install/lib/cmake/MMDeploy"
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"
ENV PATH="/root/workspace/mmdeploy/build/bin:${PATH}"
ENV PYTHONPATH="/root/workspace/mmdeploy:${PYTHONPATH}"
WORKDIR /root/workspace

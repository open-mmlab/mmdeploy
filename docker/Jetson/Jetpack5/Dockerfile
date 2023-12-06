FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
# backends
# todo : PPLNN
ARG MMDEPLOY_VERSION=main
ENV NVIDIA_VISIBLE_DEVICE all
ENV NVIDIA_DRIVER_CAPABILITIES all
# ENV CUDA_HOME="/usr/local/cuda"
# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# ENV TENSORRT_DIR="/usr/include/aarch64-linux-gnu"

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

USER root
WORKDIR /root/workspace

# install dependencies
# todo : PPLCV
RUN  apt-get update &&\ 
     apt-get install -y vim wget libspdlog-dev libssl-dev libpng-dev pkg-config libhdf5-103 libhdf5-dev --no-install-recommends

# install onnx
RUN python3 -m pip install onnx versioned-hdf5

# install onnxruntime
COPY onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl /root/workspace
# RUN wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1.whl &&\
RUN  python3 -m pip install --no-cache-dir onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl

# install mmcv
RUN git clone --branch 2.x https://github.com/open-mmlab/mmcv.git
WORKDIR /root/workspace/mmcv
RUN python3 -m pip install --no-cache-dir opencv-python==4.5.4.60
RUN MMCV_WITH_OPS=1 python3 -m pip install -e .

# build ppl.cv
COPY ppl.cv /root/workspace/ppl.cv
WORKDIR /root/workspace/ppl.cv
RUN echo "export PPLCV_DIR=/root/workspace/ppl.cv" >> ~/.bashrc
RUN ./build.sh cuda
  
# build mmdeploy
COPY mmdeploy /root/workspace/mmdeploy
# RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy &&\
# build TRT custom op
WORKDIR /root/workspace/mmdeploy
RUN mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR    &&\
    make -j$(nproc) && make install
RUN python3 -m pip install --upgrade setuptools && python3 -m pip install -e .
# build mmdeploy
RUN mkdir -p build && cd build &&\
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
RUN ln -s /usr/local/ib/python3.8/dist-packages/opencv_python.libs/* 
WORKDIR /root/workspace

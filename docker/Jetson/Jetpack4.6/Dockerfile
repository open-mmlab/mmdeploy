FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

ARG MMDEPLOY_VERSION=main
ENV NVIDIA_VISIBLE_DEVICE all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.8/site-packages/opencv-python.libs/${LD_LIBRARY_PATH}"
ENV TENSORRT_DIR="/usr/include/aarch64-linux-gnu"

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

USER root
WORKDIR /root/workspace

# install dependencies && reinstall python3.8
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA &&\
     apt-get remove python3 &&\ 
     apt-get update &&\ 
     apt-get install -y vim wget libspdlog-dev libssl-dev libpng-dev pkg-config libhdf5-100 libhdf5-dev patch --no-install-recommends\
                        python3.8 python3.8-dev python3.8-pip --no-install-recommends &&\
     python3.8 -m pip install --upgrade --no-cache-dir setuptools packaging 'Cython<3' wheel &&\
     python3.8 -m pip install --no-cache-dir --verbose wget psutil numpy &&\
     python3.8 -m pip install --upgrade --force-reinstall --no-cache-dir --verbose cmake protobuf
     python3.8 -m pip install onnx==1.10 versioned-hdf5 numpy

# build pytorch 1.10.0 for python3.8
# Hope it can works lol
# patch for https://github.com/pytorch/pytorch/issues/45323. Save here, maybe I will meet this issue.
# RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
#    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
#    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
#    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN git clone --branch v1.10.0 --depth=1 --recursive https://github.com/pytorch/pytorch /tmp/pytorch && \
    cd /tmp/pytorch && \
    wget https://gist.githubusercontent.com/dusty-nv/ce51796085178e1f38e3c6a1663a93a1/raw/4f1a0f948150c91f877aa38075835df748c81fe5/pytorch-1.10-jetpack-4.5.1.patch &&\
    patch -p1 < pytorch-1.10-jetpack-4.5.1.patch &&\
    export USE_NCCL=0 && \
    export USE_QNNPACK=0 && \
    export USE_PYTORCH_QNNPACK=0 && \
    export USE_NATIVE_ARCH=1 && \
    export USE_DISTRIBUTED=1 && \
    export USE_TENSORRT=0 && \
    python3.8 -m pip install -r requirements.txt && \
    python3.8 -m pip install --no-cache-dir scikit-build ninja && \
    python3.8 setup.py bdist_wheel && \
    cp dist/*.whl /root/workspace && \
    rm -rf /tmp/pytorch    
RUN python3 -m pip install --verbose /opt/torch*.whl

# build torchvision for python3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN git clone --branch v0.11.1 --recursive --depth=1 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout v0.11.1 && \
    python3.8 setup.py bdist_wheel && \
    cp dist/torchvision*.whl /opt && \
    rm -rf ../torchvision
RUN python3.8 -m pip install --no-cache-dir --verbose /opt/torchvision*.whl
    
# build onnxruntime for python3.8
RUN wget https://nvidia.box.com/shared/static/m9bz827ljmn771kvkjksdchmkczt3xke.whl -O onnxruntime_gpu-1.10.0-cp38-cp38-linux_aarch64.whl &&\
    python3.8 -m pip install --no-cache-dir onnxruntime_gpu-1.10.0-cp38-cp38-linux_aarch64.whl

# install mmcv
RUN git clone --branch 2.x https://github.com/open-mmlab/mmcv.git
RUN cd mmcv &&\ 
    python3.8 -m pip install --no-cache-dir opencv-python==4.5.4.60 &&\
    MMCV_WITH_OPS=1 python3 -m pip install -e .

# build ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    echo "export PPLCV_DIR=/root/workspace/ppl.cv" >> ~/.bashrc &&\
    cd ppl.cv &&\
    ./build.sh cuda
  
# build mmdeploy
RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR    &&\
    make -j$(nproc) && make install && cd .. &&\
    python3 -m pip install --upgrade setuptools &&\ 
    python3 -m pip install -e . &&\
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

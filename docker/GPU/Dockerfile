FROM nvcr.io/nvidia/tensorrt:21.04-py3                                                                                                                                        
                                                                                                                                                                              
ARG CUDA=10.2                                                                                                                                                                 
ARG PYTHON_VERSION=3.8                                                                                                                                                        
ARG TORCH_VERSION=1.8.0                                                                                                                                                       
ARG TORCHVISION_VERSION=0.9.0                                                                                                                                                 
ARG ONNXRUNTIME_VERSION=1.8.1                                                                                                                                                 
ENV FORCE_CUDA="1"                                                                                                                                                            
                                                                                                                                                                              
ENV DEBIAN_FRONTEND=noninteractive                                                                                                                                            
                                                                                                                                                                              
### update apt and install libs                                                                                                                                               
RUN apt-get update &&\                                                                                                                                                        
    apt-get install -y vim cmake libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget                                                                                      
                                                                                                                                                                              
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \                                                           
    chmod +x ~/miniconda.sh && \                                                                                                                                              
    ~/miniconda.sh -b -p /opt/conda && \                                                                                                                                      
    rm ~/miniconda.sh && \                                                                                                                                                    
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \                      
    /opt/conda/bin/conda clean -ya

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch
ENV PATH /opt/conda/bin:$PATH
#RUN pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

RUN ls /opt/conda/bin/
### install open-mim
RUN /opt/conda/bin/pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html

### git mmdetection
RUN git clone --depth=1 https://github.com/open-mmlab/mmdetection.git /root/space/mmdetection

### install mmdetection
RUN cd /root/space/mmdetection &&\
    pip3 install -r requirements.txt &&\
    python3 setup.py develop

WORKDIR /root/workspace
RUN git clone https://github.com/open-mmlab/mmclassification


### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    pip install onnxruntime-gpu

ENV PATH_TO_ONNXRUNTIME=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV PATH_TO_TENSORRT=/workspace/tensorrt

### cp trt from pip to conda
RUN cp -r /usr/local/lib/python3.8/dist-packages/tensorrt* /opt/conda/lib/python3.8/site-packages/

### update cmake to 20
RUN apt-get install -y libssl-dev &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
    tar -zxvf cmake-3.20.0.tar.gz &&\
    cd cmake-3.20.0 &&\
    ./bootstrap &&\
    make &&\
    make install

### build ppl.nn
WORKDIR /root/workspace
RUN git clone https://github.com/openppl-public/ppl.nn &&\
    cd ppl.nn &&\
    ./build.sh -DPPLNN_ENABLE_PYTHON_API=ON -DHPCC_USE_X86_64=ON -DHPCC_USE_CUDA=ON &&\
    rm -rf /tmp/pyppl-package &&\
    cp -r python/package /tmp/pyppl-package &&\
    cp -r pplnn-build/install/lib/pyppl/* /tmp/pyppl-package/pyppl &&\
    cd /tmp/pyppl-package &&\
    pip3 install .

### install mmdeploy
WORKDIR /root/workspace
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
RUN git clone https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    git submodule update --init --recursive &&\
    rm -rf build &&\
    mkdir build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS=ort .. &&\
    make -j$(nproc) &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS=trt .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    pip install -e .

### build sdk
RUN apt-get update && apt-get install libopencv-dev libspdlog-dev -y --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    ./build.sh cuda
RUN cd /root/workspace/mmdeploy &&\
    mkdir -p sdk-build && cd sdk-build &&\
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS=trt \
        -DMMDEPLOY_CODEBASES=all &&\
    cmake --build . -- -j$(nproc) && cmake --install .

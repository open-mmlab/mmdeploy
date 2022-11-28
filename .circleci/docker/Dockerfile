FROM nvcr.io/nvidia/tensorrt:21.04-py3

ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG MMCV_VERSION=1.5.0
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y libopencv-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch -c conda-forge
ENV PATH /opt/conda/bin:$PATH

### install mmcv-full
RUN /opt/conda/bin/pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html

WORKDIR /workspace

### build ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

# RUN ln -sf /opt/conda /home/circleci/project/conda
ENV TENSORRT_DIR=/workspace/tensorrt

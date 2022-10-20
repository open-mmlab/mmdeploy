FROM nvcr.io/nvidia/l4t-base:r32.7.1
ARG PYTHON_VERSION=3.6
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

WORKDIR /root/workspace
ENV FORCE_CUDA="1"
ENV CUDA_VERSION=10.2
ENV DEBIAN_FRONTEND=noninteractive

ARG USE_SRC_INSIDE=true
RUN if [ ${USE_SRC_INSIDE} ] ; \
    then \
        sed -i "s@http://.*ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi
#
# install prerequisites (many of these are for numpy)
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            python3-pip \
		  python3-dev \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir --verbose numpy

RUN wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh && \
    bash Archiconda3-0.2.3-Linux-aarch64.sh -b && \
    echo -e '\n# set environment variable for conda' >> ~/.bashrc && \
    echo ". ~/archiconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo 'export PATH=$PATH:~/archiconda3/bin' >> ~/.bashrc && \
    echo -e '\n# set environment variable for pip' >> ~/.bashrc && \
    echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc && \
    source ~/.bashrc && \
    conda --version
    # 得到默认安装的 python3 版本
    export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
    conda create -y -n mmdeploy_ python=${PYTHON_VERSION}
    conda activate mmdeploy
#
# PyTorch (for JetPack 4.4 DP)
#
#  PyTorch v1.2.0 https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl (torch-1.2.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.3.0 https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl (torch-1.3.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.4.0 https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl (torch-1.4.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.5.0 https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl (torch-1.5.0-cp36-cp36m-linux_aarch64.whl)
#

ARG PYTORCH_URL=https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl
ARG PYTORCH_WHL=torch-${TORCH_VERSION}-cp36-cp36m-linux_aarch64.whl

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --no-cache-dir --verbose ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}


#
# torchvision 0.4
#
ARG TORCHVISION_VERSION=v0.4.0
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"

RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git \
		  build-essential \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf torchvision

# note:  this was used on older torchvision versions (~0.4) to workaround a bug,
#        but has since started causing another bug as of torchvision 0.11.1
# ARG PILLOW_VERSION=pillow<7    
# pip3 install --no-cache-dir "${PILLOW_VERSION}"


# 
# upgrade cmake - https://stackoverflow.com/a/56690743
# this is needed for newer versions of torchaudio (>= v0.10)
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  software-properties-common \
		  apt-transport-https \
		  ca-certificates \
		  gnupg \
		  lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
		  
# typically --only-upgrade is used in the apt install, but cmake wasn't previously installed in this container	
# note: skipping this way for now, because having trouble pinning it to specific version (see below)
#RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
#    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
#    apt-get update && \
#    apt-cache policy cmake && \
#    apt-get install -y --no-install-recommends \
#            cmake \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get clean
    
# note:  cmake is currently pinned to 3.22.3 because of https://github.com/pytorch/pytorch/issues/74955	 
RUN pip3 install --upgrade --no-cache-dir --verbose cmake==3.22.3
RUN cmake --version


# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

    
#
# torchaudio
#
ARG TORCHAUDIO_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  pkg-config \
		  libffi-dev \
		  libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir scikit-build && \
    pip3 install --no-cache-dir ninja && \
    pip3 install --no-cache-dir --verbose pysoundfile

# note:  see https://github.com/pytorch/audio/issues/2295 for the reason for the sed commands
RUN git clone --recursive -b ${TORCHAUDIO_VERSION} https://github.com/pytorch/audio torchaudio && \
    cd torchaudio && \
    sed -i 's#  URL https://zlib.net/zlib-1.2.11.tar.gz#  URL https://zlib.net/zlib-1.2.12.tar.gz#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    sed -i 's#  URL_HASH SHA256=c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1#  URL_HASH SHA256=91844808532e5ce316b3c010929493c0244f3d37593afd6de04f71821d5136d9#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    BUILD_SOX=1 python3 setup.py install && \
    cd ../ && \
    rm -rf torchaudio


# 
# install OpenCV (with CUDA)
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

COPY scripts/opencv_install.sh /tmp/opencv_install.sh
RUN cd /tmp && ./opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}


#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN pip3 install --no-cache-dir --verbose pycuda six
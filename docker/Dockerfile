# This is a dockerfile that can be directly used for mmlab deployment,
# including onnxruntime stuff. But this will be deprivecated in the future and
# should not be open to custmers!!!
ARG OS_VERSION=18.04
ARG PYTHON_VERSION=3.7.2
ARG PYTORCH_VERSION="1.8.1"
ARG CUDA_VERSION="11.1"
ARG CUDNN_VERSION="8"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel as torch
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz /workspace/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz

ENV TENSORRT_DIR=/workspace/TensorRT-7.2.3.4
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-1.5.1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR/lib:$TENSORRT_DIR/lib
RUN conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
RUN tar -xvzf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz \
    && pip install $TENSORRT_DIR/python/tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl \
    && pip install $TENSORRT_DIR/onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl \
    && pip install $TENSORRT_DIR/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.5.1/onnxruntime-linux-x64-1.5.1.tgz \
    && tar -zxvf onnxruntime-linux-x64-1.5.1.tgz \
    && rm -rf TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz onnxruntime-linux-x64-1.5.1.tgz \
    && pip --no-cache-dir install onnxruntime==1.5.1 \
    && git clone https://github.com/open-mmlab/mmcv/ /workspace/mmcv \
    && cd mmcv \
    && MMCV_WITH_OPS=1 MMCV_WITH_TRT=1 MMCV_WITH_ORT=1 pip install -e . \
    && pip --no-cache-dir install onnx-simplifier==0.3.6 \
    && pip --no-cache-dir install onnxruntime==1.6.0 onnx==1.9.0 pycocotools terminaltables

#FROM ubuntu:${OS_VERSION} as official
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        # MMDet Requirements
        ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
        && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
ENV TENSORRT_DIR=/workspace/TensorRT-7.2.3.4
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-1.5.1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR/lib:$TENSORRT_DIR/lib
ENV PATH /opt/conda/bin:$PATH

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

RUN git clone https://github.com/open-mmlab/mmdetection \
    && cd mmdetection \
    && export ONNX_BACKEND=MMCVTensorRT \
    && export PYTHONPATH=$PYTHONPATH:pwd \
    && python tools/deployment/pytorch2onnx.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth --input-img demo/demo.jpg --test-img  demo/demo.jpg --dynamic-export
#    && python tools/deployment/onnx2tensorrt.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py tmp.onnx --trt-file tmp.trt --verify --min-shape 300 300 --max-shape 1600 1600
CMD /bin/bash

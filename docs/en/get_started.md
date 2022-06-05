# Get Started

MMDeploy provides useful tools for deploying OpenMMLab models to various platforms and devices.

With the help of them, you can not only do model deployment using our pre-defined pipelines but also customize your own deployment pipeline.

In the following chapters, we will describe the general routine and demonstrate a "hello-world" example - deploying Faster R-CNN model from MMDetection to NVIDIA TensorRT.

## Introduction

In MMDeploy, the deployment pipeline can be illustrated by a sequential modules, i.e., Model Converter, MMDeploy Model and Inference SDK.

![deploy-pipeline](https://user-images.githubusercontent.com/4560679/171416470-8020f967-39de-4c19-ad46-a4197c970874.png)

### Model Converter

模型转换的主要功能是把输入的模型格式，转换为目标设备的推理引擎所要求的模型格式。

目前，MMDeploy 可以把 PyTorch 模型转换为 ONNX、TorchScript 等和设备无关的 IR 模型。也可以将 ONNX 模型转换为推理后端模型。两者相结合，可实现端到端的模型转换，也就是从训练端到生产端的一键式部署。

### MMDeploy Model

模型转换结果的集合。它不仅包括后端模型，还包括模型的元信息。这些信息将用于推理 SDK 中。

### Inference SDK

封装了模型的前处理、网络推理和后处理过程。对外提供多语言的模型推理接口。

## prerequisites

In order to do an end-to-end model deployment, MMDeploy requires Python 3.6+ and PyTorch 1.5+.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
export PYTHON_VERSION=3.7
conda create --name mmdeploy python=${PYTHON_VERSION} -y
conda activate mmdeploy
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
  export PYTORCH_VERSION=1.8.0
  export TORCHVISION_VERSION=0.9.0
  export CUDA_VERSION=11.1
  conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -c conda-forge
  ```

On CPU platforms:

```shell
export PYTORCH_VERSION=1.8.0
export TORCHVISION_VERSION=0.9.0
conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices installing MMDeploy.

***Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv).
```shell
  export MMCV_VERSION=1.5.0
  export CUDA_STRING="${CUDA_VERSION/./""}"
  python -m pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_STRING}/torch${PYTORCH_VERSION}/index.html
  ```

**Step 1.** Install MMDeploy.
Since v0.5.0, MMDeploy provides prebuilt packages, which can be found from [here](https://github.com/open-mmlab/mmdeploy/releases).
You can download them according to your target platform and device.

Take the MMDeploy-TensorRT package on NVIDIA for example:

  ```shell
  export MMDEPLOY_VERSION=0.5.0
  export TENSORRT_VERSION=8.2.3.0

  wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.5.0/mmdeploy-v0.5.0-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}.tar.gz
  tar -zxvf mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}.tar.gz
  cd mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}
  python -m pip install dist/mmdeploy-*-py${PYTHON_STRING}*.whl
  python -m pip install sdk/python/mmdeploy_python-*-cp${PYTHON_STRING}*.whl
  export LD_LIBRARY_PATH=$(pwd)/sdk/lib:$LD_LIBRARY_PATH
  cd ..
  ```

  ```{note}
  If MMDeploy prebuilt package doesn meet your target platforms or devices, please build MMDeploy from its source by following the build documents
  ```

**step 2**： Install the inference backend

Based on the above MMDeploy-TensorRT package, we need to download and install [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) (including [cuDNN](https://developer.nvidia.com/cudnn)).

Take TensorRT 8.2.3.0 for example:

  ```shell
  export TENSORRT_VERSION=8.2.3.0
  CUDA_MAJOR="${CUDA_VERSION/\.*/""}"

  # !!! Download tensorrt package from NVIDIA that matches your CUDA Toolkit version to the current working directory
  tar -zxvf TensorRT-${TENSORRT_VERSION}*cuda-${CUDA_MAJOR}*.tar.gz
  python -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_STRING}*.whl
  export TENSORRT_DIR=$(pwd)/TensorRT-${TENSORRT_VERSION}
  export CUDNN_VERSION=`ls TensorRT-${TENSORRT_VERSION}*cuda-${CUDA_MAJOR}*.tar.gz | sed "s/.*cudnn/v/g" | sed "s/\.tar\.gz//g"`

  # !!! Download cuDNN package from NVIDIA that matches your CUDA Toolkit and TensorRT version to the current working directory
  tar -zxvf cudnn-${CUDA_MAJOR}.*-linux-x64-${CUDNN_VERSION}*.tgz
  export CUDNN_DIR=$(pwd)/cuda
  export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
  ```

For the installation of all inference backends supported by MMDeploy right now, please refer to:

- [ONNX Runtime](05-supported-backends/onnxruntime.md)
- [TensorRT](05-supported-backends/tensorrt.md)
- [PPL.NN](05-supported-backends/pplnn.md)
- [ncnn](05-supported-backends/ncnn.md)
- [OpenVINO](05-supported-backends/openvino.md)
- [LibTorch](05-supported-backends/torchscript.md)

## Convert Model


## Inference Model

### Inference by Model Converter


### Inference by SDK

#### Python API

#### C API

#### C# API

#### Java API


## Evaluate Model

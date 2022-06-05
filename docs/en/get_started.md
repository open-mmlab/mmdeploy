# Get Started

MMDeploy provides useful tools for deploying OpenMMLab models to various platforms and devices.

With the help of them, you can not only do model deployment using our pre-defined pipelines but also customize your own deployment pipeline.

In the following chapters, we will describe the general routine and demonstrate a "hello-world" example - deploying Faster R-CNN model from MMDetection to NVIDIA TensorRT.

## Introduction

In MMDeploy, the deployment pipeline can be illustrated by a sequential modules, i.e., Model Converter, MMDeploy Model and Inference SDK.

![deploy-pipeline](https://user-images.githubusercontent.com/4560679/171416470-8020f967-39de-4c19-ad46-a4197c970874.png)

### Model Converter

Model Converter aims at converting training models from OpenMMLab into backend models that can be run on target devices.

Currently, MMDeploy is able to transform PyTorch model into IR model, i.e., ONNX, TorchScript, as well as convert IR model to backend model.
By combining them together, we can achieve one-click **end-to-end** model deployment.

### MMDeploy Model

MMDeploy Model is the result set exported by Model Converter.
Beside the backend models, it also includes the model meta info, which will be performed by Inference SDK.

### Inference SDK

Inference SDK is developed by C/C++, wrapping the preprocessing, model forward and postprocessing modules in model inference.
It supports FFI such as C, C++, Python, C#, Java and so on.

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

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv).
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

**step 2.** Install the inference backend

Based on the above MMDeploy-TensorRT package, we need to download and install [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) (including [cuDNN](https://developer.nvidia.com/cudnn)).

The following shows an example of installing TensorRT 8.2.3.0:

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

After the installation, you can enjoy the model deployment journey starting from converting PyTorch model to backend model.

Based on the above settings, we provide an example to convert the Faster R-CNN in [MMDetection](https://github.com/open-mmlab/mmdetection) to TensorRT as below:

```shell
# clone mmdeploy repo. We are going to use the pre-defined pipeline config from the source code
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
python -m pip install mmdeploy/requirements/runtime.txt
export MMDEPLOY_DIR=$(pwd)/mmdeploy

# clone mmdetection repo. We have to use the config file to build PyTorch nn module
python -m pip install mmdet==2.24.0
git clone https://github.com/open-mmlab/mmdetection.git
export MMDET_DIR=$(pwd)/mmdetection

# download Faster R-CNN checkpoint
export CHECKPOINT_DIR=$(pwd)/checkpoints
wget -P ${CHECKPOINT_DIR} https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# set working directory, where the mmdeploy model is saved
export WORK_DIR=$(pwd)/mmdeploy_models

# run the command to start model conversion
python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${MMDET_DIR}/demo/demo.jpg \
    --work-dir ${WORK_DIR}/faster-rcnn \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` does everything you need to convert a model. Read [how_to_convert_model](./02-how-to-run/how_to_convert_model.md) for more details.
The converted model and its meta info will be found in the path specified by `--work-dir`.
And they make up of MMDeploy Model that can be fed to MMDeploy SDK to do model inference.

`detection_tensorrt_dynamic-320x320-1344x1344.py` is a config file that contains all arguments you need to customize the conversion pipeline. The name is formed as:

```bash
<task name>_<backend>-[backend options]_<dynamic support>.py
```

If you want to customize the conversion pipeline, you can edit the config file by following [this](./02-how-to-run/how_to_write_config.md) tutorial.

## Inference Model

### Inference by Model Converter



### Inference by SDK

You can use SDK API to do model inference with the generated mmdeploy model by Model Converter.

In the following section, we will provide examples of deploying the converted Faster R-CNN model in different FFI.

#### Python API

```python
from mmdeploy_python import Detector
import os
import cv2

# get mmdeploy model path of faster r-cnn
model_path = '/'.join((os.getenv('WORK_DIR'), '/faster-rcnn'));
# use mmdetection demo image as an input image
image_path = '/'.join((os.getenv('MMDET_DIR'), 'demo/demo.jpg'));

img = cv2.imread(image_path)
detector = Detector(model_path, 'cuda', 0)
bboxes, labels, _ = detector([img])[0]

indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```
You can find more examples from [here](https://github.com/open-mmlab/mmdeploy/demo/python).

#### C API

#### C# API

#### Java API


## Evaluate Model

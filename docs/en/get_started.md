# Get Started

MMDeploy provides useful tools for deploying OpenMMLab models to various platforms and devices.

With the help of them, you can not only do model deployment using our pre-defined pipelines but also customize your own deployment pipeline.

In the following chapters, we will describe the general routine and demonstrate a "hello-world" example - deploying Faster R-CNN model from [MMDetection](https://github.com/open-mmlab/mmdetection) to NVIDIA TensorRT.

## Introduction

In MMDeploy, the deployment pipeline can be illustrated by a sequential modules, i.e., Model Converter, MMDeploy Model and Inference SDK.

![deploy-pipeline](https://user-images.githubusercontent.com/4560679/172306700-31b4c922-2f04-42ed-a1d6-c360f2f3048c.png)

### Model Converter

Model Converter aims at converting training models from OpenMMLab into backend models that can be run on target devices.
It is able to transform PyTorch model into IR model, i.e., ONNX, TorchScript, as well as convert IR model to backend model. By combining them together, we can achieve one-click **end-to-end** model deployment.

### MMDeploy Model

MMDeploy Model is the result package exported by Model Converter.
Beside the backend models, it also includes the model meta info, which will be used by Inference SDK.

### Inference SDK

Inference SDK is developed by C/C++, wrapping the preprocessing, model forward and postprocessing modules in model inference.
It supports FFI such as C, C++, Python, C#, Java and so on.

## Prerequisites

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
export PYTHON_VERSION=3.7
export PYTHON_STRING="${PYTHON_VERSION/./""}"

wget https://github.com/open-mmlab/mmdeploy/releases/download/v${MMDEPLOY_VERSION}/mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}.tar.gz
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

Based on the above MMDeploy-TensorRT package, we need to download and install [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar), including [cuDNN](https://developer.nvidia.com/cudnn).

**Be aware that TensorRT version and cuDNN version must matches your CUDA Toolkit version**

The following shows an example of installing TensorRT 8.2.3.0 and cuDNN 8.2:

```shell
export TENSORRT_VERSION=8.2.3.0
CUDA_MAJOR="${CUDA_VERSION/\.*/""}"

# !!! Download tensorrt package from NVIDIA that matches your CUDA Toolkit version to the current working directory
tar -zxvf TensorRT-${TENSORRT_VERSION}*cuda-${CUDA_MAJOR}*.tar.gz
python -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_STRING}*.whl
python -m pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-${TENSORRT_VERSION}
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH

# !!! Download cuDNN package from NVIDIA that matches your CUDA Toolkit and TensorRT version to the current working directory
tar -zxvf cudnn-${CUDA_MAJOR}.*-linux-x64*.tgz
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

In the next chapters, we are going to present our 'Hello, world' example based on the above settings.

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
python -m pip install -r mmdeploy/requirements/runtime.txt
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
    --work-dir ${WORK_DIR} \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` does everything you need to convert a model. Read [how_to_convert_model](./02-how-to-run/convert_model.md) for more details.
The converted model and its meta info will be found in the path specified by `--work-dir`.
And they make up of MMDeploy Model that can be fed to MMDeploy SDK to do model inference.

`detection_tensorrt_dynamic-320x320-1344x1344.py` is a config file that contains all arguments you need to customize the conversion pipeline. The name is formed as:

```bash
<task name>_<backend>-[backend options]_<dynamic support>.py
```

If you want to customize the conversion pipeline, you can edit the config file by following [this](./02-how-to-run/write_config.md) tutorial.

## Inference Model

After model conversion, we can perform inference both by Model Converter and Inference SDK.

The former is developed by Python, while the latter is mainly written by C/C++.

### Inference by Model Converter

Model Converter provides a unified API named as `inference_model` to do the job, making all inference backends API transparent to users.
Take the previous converted Faster R-CNN tensorrt model for example,

```python
from mmdeploy.apis import inference_model
import os

model_cfg = os.getenv('MMDET_DIR') + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
deploy_cfg = os.getenv('MMDEPLOY_DIR') + '/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
backend_files = [os.getenv('WORK_DIR') + '/end2end.engine']

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

The data type and data layout is exactly the same with the OpenMMLab PyTorch model inference results.

```{note}
You can certainly use the infernce backend API directly to do inference. But since MMDeploy has being developed several custom operators, it's necessary to load them first before calling the infernce backend API.
```

### Inference by SDK

You can use SDK API to do model inference with the mmdeploy model generated by Model Converter.

In the following section, we will provide examples of deploying the converted Faster R-CNN model talked above with different FFI.

#### Python API

```python
from mmdeploy_python import Detector
import os
import cv2

# get mmdeploy model path of faster r-cnn
model_path = os.getenv('WORK_DIR')
# use mmdetection demo image as an input image
image_path = '/'.join((os.getenv('MMDET_DIR'), 'demo/demo.jpg'))

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

You can find more examples from [here](https://github.com/open-mmlab/mmdeploy/tree/master/demo/python).

```{note}
If you build MMDeploy from the source, please add ${MMDEPLOY_DIR}/build/lib to the environment variable PYTHONPATH.
Otherwise, you will run into an error like ’ModuleNotFoundError: No module named 'mmdeploy_python'
```

#### C API

Using SDK C API should follow next pattern,

```mermaid
graph LR
  A[create inference handle] --> B(read image)
  B --> C(apply handle)
  C --> D[deal with inference result]
  D -->E[destroy result buffer]
  E -->F[destroy handle]
```

Now let's apply this procedure on the above Faster R-CNN model.

```C++
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "detector.h"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;

  // get mmdeploy model path of faster r-cnn
  std::string model_path = std::getenv("WORK_DIR");
  // use mmdetection demo image as an input image
  std::string image_path = std::getenv("MMDET_DIR") + "/demo/demo.jpg";

  // create inference handle
  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector);
  assert(status == MM_SUCCESS);

  // read image
  cv::Mat img = cv::imread(image_path);
  assert(img.data);

  // apply handle and get the inference result
  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};
  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  assert (status == MM_SUCCESS);

  // deal with the result. Here we choose to visualize it
  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite('output_detection.png', img);

  // destroy result buffer
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  // destroy inference handle
  mmdeploy_detector_destroy(detector);
  return 0;
}
```

When you build this example, try to add MMDeploy package in your CMake project as following. Then pass `-DMMDeploy_DIR` to cmake, which indicates the path where `MMDeployConfig.cmake` locates. You can find it in the prebuilt package.

```Makefile
find_package(MMDeploy REQUIRED)
mmdeploy_load_static(${YOUR_AWESOME_TARGET} MMDeployStaticModules)
mmdeploy_load_dynamic(${YOUR_AWESOME_TARGET} MMDeployDynamicModules)
target_link_libraries(${YOUR_AWESOME_TARGET} PRIVATE MMDeployLibs)
```

For more SDK C API usages, please read these [samples](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc).

#### C# API

Due to limitations on space, we will not present a specific example. But you can find all of them [here](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csharp).

## Evaluate Model

You can test the performance of deployed model using `tool/test.py`. For example,

```shell
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

```{note}
Regarding the --model option, it represents the converted engine files path when using Model Converter to do performance test. But when you try to test the metrics by Inference SDK, this option refers to the directory path of MMDeploy Model.
```

You can read [how to evaluate a model](02-how-to-run/how_to_evaluate_a_model.md) for more details.

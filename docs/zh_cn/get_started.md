# 操作概述

MMDeploy提供了一系列工具，帮助您更轻松的将OpenMMLab下的算法部署到各种设备与平台上。您可以使用我们设计的流程一“部”到位，也可以定制您自己的转换流程。这份指引将会向您展示MMDeploy的基本使用方式。

## 简介

MMDeploy 定义的模型部署流程，如下图所示：
![deploy-pipeline](https://user-images.githubusercontent.com/4560679/171416470-8020f967-39de-4c19-ad46-a4197c970874.png)

### 模型转换（Model Converter）

模型转换的主要功能是把输入的模型格式，转换为目标设备的推理引擎所要求的模型格式。
目前，MMDeploy 可以把 PyTorch 模型转换为 ONNX、TorchScript 等和设备无关的 IR 模型。也可以将 ONNX 模型转换为推理后端模型。两者相结合，可实现端到端的模型转换，也就是从训练端到生产端。

### MMDeploy 模型（MMDeploy Model）

模型转换结果的集合。它不仅包括后端模型，还包括模型的元信息。这些信息将用于推理 SDK 中。

### 推理 SDK（Inference SDK）

封装了模型的前处理、网络推理和后处理过程。对外提供多语言的模型推理接口。

## 准备工作

对于端到端的模型转换和推理，需要按照如下步骤配置环境：

1. 创建 conda 环境，并安装 PyTorch。因为 Model Converter 的 torch2onnx 功能依赖它
2. 安装 mmcv-full
3. 安装 mmdeploy。有以下两种方式：
   - 方式 1：安装预编译包。根据目标软硬件平台，从[这里]()下载并安装 MMDeploy 预编译包
   - 方式 2：[源码安装](01-how-to-build/build_from_source.md)
4. 安装预编译包要求的推理后端

    以 Ubuntu 20.04 NVIDIA GPU（CUDA Toolkit 11.1）为例，我们可以通过以下脚本完成准备工作。

    ```shell
    PYTHON_VERSION=3.7
    PYTORCH_VERSION=1.8.0
    TORCHVISION_VERSION=0.9.0
    CUDA_VERSION=11.1
    MMCV_VERSION=1.5.0
    MMDEPLOY_VERSION=0.5.0
    TENSORRT_VERSION=8.2.3.0

    PYTHON_STRING="${PYTHON_VERSION/./""}"
    CUDA_STRING="${CUDA_VERSION/./""}"

    # 1. 创建 conda 环境
    conda create -n mmdeploy python=${PYTHON_VERSION} -y
    conda activate mmdeploy

    # 2. 安装 PyTorch
    conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -c conda-forge

    # 3. 安装 mmcv-full
    python -m pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_STRING}/torch${PYTORCH_VERSION}/index.html

    # 4. 安装 mmdeploy 预编译包
    # wget ****
    tar -zxvf mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}.tar.gz
    cd mmdeploy-${MMDEPLOY_VERSION}-linux-x86_64-cuda${CUDA_VERSION}-tensorrt${TENSORRT_VERSION}
    python -m pip install dist/mmdeploy-*-py${PYTHON_STRING}*.whl
    python -m pip install sdk/python/mmdeploy_python-*-cp${PYTHON_STRING}*.whl
    export LD_LIBRARY_PATH=$(pwd)/sdk/lib:$LD_LIBRARY_PATH
    cd ..

    # 5. 安装预编译包要求的推理后端，以及其他依赖包
    # !!! 从 NVIDIA 官网下载 tensorrt
    # cd /the/path/of/tensorrt/tar/gz/file
    tar -zxvf TensorRT-${TENSORRT_VERSION}*.tar.gz
    python -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_STRING}*.whl
    export TENSORRT_DIR=$(pwd)/TensorRT-${TENSORRT_VERSION}

    # !!! 从 NVIDIA 官网下载与 cuda toolkit，tensorrt 匹配的 cudnn
    # cd /the/path/of/cudnn/tgz/file
    tar -zxvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
    export CUDNN_DIR=$(pwd)/cuda
    export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
    ```

在接下来的章节中，我们均以此环境为基础，演示 MMDeploy 的功能。

**如果 MMDeploy 没有您所需要的目标软硬件平台的预编译包，请参考[源码安装文档](01-how-to-build/build_from_source.md)正确安装和配置**

## 模型转换

在准备工作就绪后，我们可以使用 MMDeploy 中的工具 `deploy.py`，将 OpenMMLab 的 PyTorch 模型转换成推理后端支持的格式。
以 [MMDetection](https://github.com/open-mmlab/mmdetection) 中的 `Faster-RCNN` 为例，我们可以使用如下命令，将 PyTorch 模型转换成可部署在 NVIDIA GPU 上的 TenorRT 模型：

```shell
# 克隆 mmdeploy 仓库。转换时，需要使用 mmdeploy 仓库中的配置文件，建立转换流水线
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
python -m pip install mmdeploy/requirements/runtime.txt
export MMDEPLOY_DIR=$(pwd)/mmdeploy

# 克隆 mmdetection 仓库。转换时，需要使用 mmdetection 仓库中的模型配置文件，构建 PyTorch nn module
python -m pip install mmdet==2.24.0
git clone https://github.com/open-mmlab/mmdetection.git
export MMDET_DIR=$(pwd)/mmdetection

# 下载 Faster R-CNN 模型权重
export CHECKPOINT_DIR=$(pwd)/checkpoints
wget -P ${CHECKPOINT_DIR} https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# 执行转换命令，实现端到端的转换
python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${MMDET_DIR}/demo/demo.jpg \
    --work-dir ${MMDEPLOY_DIR}/mmdeploy_model/faster-rcnn \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` 是一个方便模型转换的工具。可以阅读 [如何转换模型](./02-how-to-run/convert_model.md) 了解更多细节。
`detection_tensorrt_dynamic-320x320-1344x1344.py` 是一个参数配置文件。该文件的命名遵循如下规则：

```bash
<任务名>_<推理后端>-[后端特性]_<动态模型支持>.py
```

可以很容易的通过文件名来确定最适合的那个配置文件。如果您希望定制自己的转换配置，可以参考[如何编写配置文件](./02-how-to-run/write_config.md)修改参数。

## 模型推理

### 使用推理后端 API

因为 MMDeploy 拥有自定义算子，所以在使用后端 API 推理前，需要先加载 MMDeploy 自定义算子库：

```shell
```

### 使用 Model Converter 的推理 API

Model Converter 对推理后端的 API 进行了统一封装：
```python
from mmdeploy.apis import inference_model

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

`inference_model`会创建一个对后端模型的封装，通过该封装进行推理。推理的结果会保持与 OpenMMLab 中原模型同样的格式。

### 使用 MMDeploy SDK API

以上文中转出的 Faster R-CNN TensorRT 模型为例，接下来的章节将介绍如何使用 SDK 的 FFI 进行模型推理。

#### Python API

```python
import mmdeploy_python
import sys
import cv2

device_name = "cuda";
device_id = 0;
model_path = "mmdeploy/mmdeploy_model/faster-rcnn";
image_path = "mmdetection/demo/demo.jpg";

img = cv2.imread(image_path)
detector = mmdeploy_python.Detector(model_path, device_name, device_id)
result = detector([img])
```

更多模型的 SDK Python API 应用样例，请查阅 mmdeploy/demo/python。

#### C API

使用 C API 进行模型推理的流程符合下面的模式：

创建推理句柄 -> 读取图像 -> 应用句柄进行推理 -> 处理推理结果 -> 销毁结果 -> 销毁推理句柄

我们以检测器（mmdeploy/demo/csrc/object_detection.cpp）为例，来说明这个流程是如何实现的：

```C++

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "detector.h"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;
   // MMDeploy Model directory
  const char* model_path = "mmdeploy/mmdeploy_model/faster-rcnn";
  const char* image_path = "mmdetection/demo/demo.jpg";

  // create detector handle
  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector);
  assert(status == MM_SUCCESS);

  cv::Mat img = cv::imread(image_path);
  assert(img.data);

  // model inference
  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};
  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  assert (status == MM_SUCCESS);

  // visualize result
  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
     // skip detections less than specified score threshold
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imshow("faster-rcnn", img);
  cv::waitKey(0);

  // destroy result buffer and handle
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  mmdeploy_detector_destroy(detector);
  return 0;
}
```

在您的项目CMakeLists中，增加：

```Makefile
find_package(MMDeploy REQUIRED)
mmdeploy_load_static(${YOUR_AWESOME_TARGET} MMDeployStaticModules)
mmdeploy_load_dynamic(${YOUR_AWESOME_TARGET} MMDeployDynamicModules)
target_link_libraries(${YOUR_AWESOME_TARGET} PRIVATE MMDeployLibs)
```

编译时，使用 -DMMDeploy_DIR，传入MMDeloyConfig.cmake所在的路径。它在预编译包中的sdk/lib/cmake/MMDeloy下。
更多模型的 SDK C API 应用样例，请查阅 mmdeploy/demo/csrc。

#### C# API

请参考 mmdeploy/demo/csharp/* 下的例子，了解 SDK C# API 的用法。

#### Java API

请参考 mmdeploy/demo/java/* 下的例子，了解 SDK Java API 的用法。

## 模型精度评估

为了测试部署模型的精度，推理效率，我们提供了 `tools/test.py` 来帮助完成相关工作。以上文中 Faster-RCNN 的 TensorRT 模型为例：

```bash
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

请阅读 [如何进行模型评估](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_evaluate_a_model.html) 了解关于 `tools/test.py` 的使用细节。

# 操作概述

MMDeploy 提供了一系列工具，帮助您更轻松的将 OpenMMLab 下的算法部署到各种设备与平台上。

您可以使用我们设计的流程一“部”到位，也可以定制您自己的转换流程。

在接下来的章节中，我们将会向您展示 MMDeploy 的模型部署方式。并在 NVIDIA 设备上，以 [MMDetection](https://github.com/open-mmlab/mmdetection) Faster R-CNN 模型为例，演示 MMDeploy 的基本使用方法。

## 流程简介

MMDeploy 定义的模型部署流程，如下图所示：
![deploy-pipeline](https://user-images.githubusercontent.com/4560679/172306700-31b4c922-2f04-42ed-a1d6-c360f2f3048c.png)

### 模型转换（Model Converter）

模型转换的主要功能是把输入的模型格式，转换为目标设备的推理引擎所要求的模型格式。

目前，MMDeploy 可以把 PyTorch 模型转换为 ONNX、TorchScript 等和设备无关的 IR 模型。也可以将 ONNX 模型转换为推理后端模型。两者相结合，可实现端到端的模型转换，也就是从训练端到生产端的一键式部署。

### MMDeploy 模型（MMDeploy Model）

模型转换结果的集合。它不仅包括后端模型，还包括模型的元信息。这些信息将用于推理 SDK 中。

### 推理 SDK（Inference SDK）

封装了模型的前处理、网络推理和后处理过程。对外提供多语言的模型推理接口。

## 准备工作

对于端到端的模型转换和推理，MMDeploy 依赖 Python 3.6+ 以及 PyTorch 1.5+。

**第一步**：从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda

**第二步**：创建并激活 conda 环境

```shell
export PYTHON_VERSION=3.7
conda create --name mmdeploy python=${PYTHON_VERSION} -y
conda activate mmdeploy
```

**第三步**: 参考[官方文档](https://pytorch.org/get-started/locally/)并安装 PyTorch

Model Converter 的 torch2onnx 功能依赖它。

在 GPU 环境下（这里我们以 Ubuntu 18.04 CUDA 11.1 为基础），您可以使用如下方式安装 PyTorch 1.8：

```shell
export PYTHON_VERSION=3.7
export PYTORCH_VERSION=1.8.0
export TORCHVISION_VERSION=0.9.0
export CUDA_VERSION=11.1

conda create -n mmdeploy python=${PYTHON_VERSION} -y
conda activate mmdeploy

conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch -c conda-forge
```

在 CPU 环境下，您可以执行：

```shell
export PYTORCH_VERSION=1.8.0
export TORCHVISION_VERSION=0.9.0
conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cpuonly -c pytorch
```

## 安装 MMDeploy

**第一步**: 安装 mmcv-full

```shell
export MMCV_VERSION=1.5.0
export CUDA_STRING="${CUDA_VERSION/./""}"

python -m pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_STRING}/torch${PYTORCH_VERSION}/index.html
```

**第二步**: 安装 MMDeploy

从 v0.5.0 之后，MMDeploy 开始提供预编译包。您可以根据目标软硬件平台，从[这里](https://github.com/open-mmlab/mmdeploy/releases)选择并下载预编译包。

在 NVIDIA 设备上，我们推荐使用 MMDeploy-TensoRT 预编译包：

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
如果 MMDeploy 没有您所需要的目标软硬件平台的预编译包，请参考源码安装文档，正确安装和配置
```

**第三步**： 安装预编译包要求的推理后端

在本例中，我们需要安装 TensorRT（含 cuDNN）推理引擎。因在 NVIDIA 官网下载软件包，必须要登录认证，所以请预先登录并下载所需的 [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) 和 [cuDNN](https://developer.nvidia.com/cudnn)。**请注意： TensorRT 版本、cuDNN 版本要和 CUDA 版本匹配**

下载完毕后，您可以参考如下方法安装。这里，我们以 TensorRT 8.2.3.0、cuDNN 8.2 为例：

```shell
export TENSORRT_VERSION=8.2.3.0
CUDA_MAJOR="${CUDA_VERSION/\.*/""}"

# !!! 从 NVIDIA 官网下载 与 cuda toolkit 匹配的 tensorrt 到当前的工作目录
tar -zxvf TensorRT-${TENSORRT_VERSION}*cuda-${CUDA_MAJOR}*.tar.gz
python -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_STRING}*.whl
python -m pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-${TENSORRT_VERSION}
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH


# !!! 从 NVIDIA 官网下载与 cuda toolkit，tensorrt 匹配的 cudnn 到当前的工作目录
tar -zxvf cudnn-${CUDA_MAJOR}.*-linux-x64*.tgz
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

在接下来的章节中，我们均以此环境为基础，演示 MMDeploy 的功能。

目前，对于 MMDeploy 支持各种推理后端的安装方法，您可以查阅以下文档：

- [ONNX Runtime](../en/05-supported-backends/onnxruntime.md)
- [TensorRT](../en/05-supported-backends/tensorrt.md)
- [PPL.NN](../en/05-supported-backends/pplnn.md)
- [ncnn](../en/05-supported-backends/ncnn.md)
- [OpenVINO](../en/05-supported-backends/openvino.md)
- [LibTorch](../en/05-supported-backends/torchscript.md)

## 模型转换

在准备工作就绪后，我们可以使用 MMDeploy 中的工具 `deploy.py`，将 OpenMMLab 的 PyTorch 模型转换成推理后端支持的格式。

以 [MMDetection](https://github.com/open-mmlab/mmdetection) 中的 `Faster R-CNN` 为例，我们可以使用如下命令，将 PyTorch 模型转换成可部署在 NVIDIA GPU 上的 TenorRT 模型：

```shell
# 克隆 mmdeploy 仓库。转换时，需要使用 mmdeploy 仓库中的配置文件，建立转换流水线
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
python -m pip install -r mmdeploy/requirements/runtime.txt
export MMDEPLOY_DIR=$(pwd)/mmdeploy

# 克隆 mmdetection 仓库。转换时，需要使用 mmdetection 仓库中的模型配置文件，构建 PyTorch nn module
python -m pip install mmdet==2.24.0
git clone https://github.com/open-mmlab/mmdetection.git
export MMDET_DIR=$(pwd)/mmdetection

# 下载 Faster R-CNN 模型权重
export CHECKPOINT_DIR=$(pwd)/checkpoints
wget -P ${CHECKPOINT_DIR} https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# 设置工作路径
export WORK_DIR=$(pwd)/mmdeploy_models/faster-rcnn

# 执行转换命令，实现端到端的转换
python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${MMDET_DIR}/demo/demo.jpg \
    --work-dir ${WORK_DIR} \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` 是一个方便模型转换的工具。您可以阅读 [如何转换模型](./02-how-to-run/convert_model.md) 了解更多细节。

`detection_tensorrt_dynamic-320x320-1344x1344.py` 是一个参数配置文件。该文件的命名遵循如下规则：

```bash
<任务名>_<推理后端>-[后端特性]_<动态模型支持>.py
```

可以很容易的通过文件名来确定最适合的那个配置文件。如果您希望定制自己的转换配置，可以参考[如何编写配置文件](./02-how-to-run/write_config.md)修改参数。

## 模型推理

在转换完成后，您既可以使用 Model Converter 进行推理，也可以使用 Inference SDK。前者使用 Python 开发，后者主要使用 C/C++ 开发。

### 使用 Model Converter 的推理 API

Model Converter 屏蔽了推理后端接口的差异，对其推理 API 进行了统一封装，接口名称为 `inference_model`。

以上文中 Faster R-CNN 的 TensorRT 模型为例，您可以使用如下方式进行模型推理工作：

```python
from mmdeploy.apis import inference_model
import os

model_cfg = os.getenv('MMDET_DIR') + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
deploy_cfg = os.getenv('MMDEPLOY_DIR') + '/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
backend_files = os.getenv('WORK_DIR') + '/end2end.engine'

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

`inference_model`会创建一个对后端模型的封装，通过该封装进行推理。推理的结果会保持与 OpenMMLab 中原模型同样的格式。

```{note}
MMDeploy 转出的后端模型，您可以直接使用后端 API 进行推理。不过，因为 MMDeploy 拥有 TensorRT、ONNX Runtime 等自定义算子，
您需要先加载对应的自定义算子库，然后再使用后端 API。
```

### 使用推理 SDK

您也可以使用 MMDeploy SDK 进行推理。以上文中转出的 Faster R-CNN TensorRT 模型为例，接下来的章节将介绍如何使用 SDK 的 FFI 进行模型推理。

#### Python API

```python
from mmdeploy_python import Detector
import os
import cv2

# 获取转换后的 mmdeploy model 路径
model_path = os.getenv('WORK_DIR')
# 从 mmdetection repo 中，获取 demo.jpg 路径
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

更多模型的 SDK Python API 应用样例，请查阅[这里](https://github.com/open-mmlab/mmdeploy/tree/master/demo/python)。

```{note}
如果您使用源码安装方式, 请把 ${MMDEPLOY_DIR}/build/lib 加入到环境变量 PYTHONPATH 中。
否则会遇到错误’ModuleNotFoundError: No module named 'mmdeploy_python'
```

#### C API

使用 C API 进行模型推理的流程符合下面的模式：

```mermaid
graph LR
  A[创建推理句柄] --> B(读取图像)
  B --> C(应用句柄进行推理)
  C --> D[处理推理结果]
  D -->E[销毁结果]
  E -->F[销毁推理句柄]
```

以下是这个流程的具体应用过程：

```C++
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "detector.h"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;

  // 获取转换后的 mmdeploy model 路径
  std::string model_path = std::getenv("WORK_DIR");
  // 从 mmdetection repo 中，获取 demo.jpg 路径
  std::string image_path = std::getenv("MMDET_DIR") + "/demo/demo.jpg";

  // 创建推理句柄
  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector);
  assert(status == MM_SUCCESS);

  // 读取图像
  cv::Mat img = cv::imread(image_path);
  assert(img.data);

  // 应用句柄进行推理
  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};
  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  assert (status == MM_SUCCESS);

  // 处理推理结果: 此处我们选择可视化推理结果
  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite('output_detection.png', img);

  // 销毁结果
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  // 销毁推理句柄
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
更多模型的 SDK C API 应用样例，请查阅[此处](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc)。

#### C# API

因篇幅所限，本文不展示具体案例。请参考[这里](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csharp)，了解 SDK C# API 的用法。

## 模型精度评估

为了测试部署模型的精度，推理效率，我们提供了 `tools/test.py` 来帮助完成相关工作。以上文中的部署模型为例：

```bash
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

```{note}
关于 --model 选项，当使用 Model Converter 进行推理时，它代表转换后的推理后端模型的文件路径。而当使用 SDK 测试模型精度时，该选项表示 MMDeploy Model 的路径.
```

请阅读 [如何进行模型评估](./02-how-to-run/profile_model.md) 了解关于 `tools/test.py` 的使用细节。

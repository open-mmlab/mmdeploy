# 操作概述

MMDeploy 提供了一系列工具，帮助您更轻松的将 OpenMMLab 下的算法部署到各种设备与平台上。

您可以使用我们设计的流程一“部”到位，也可以定制您自己的转换流程。

## 流程简介

MMDeploy 定义的模型部署流程，如下图所示：
![deploy-pipeline](https://user-images.githubusercontent.com/4560679/172306700-31b4c922-2f04-42ed-a1d6-c360f2f3048c.png)

### 模型转换（Model Converter）

模型转换的主要功能是把输入的模型格式，转换为目标设备的推理引擎所要求的模型格式。

目前，MMDeploy 可以把 PyTorch 模型转换为 ONNX、TorchScript 等和设备无关的 IR 模型。也可以将 ONNX 模型转换为推理后端模型。两者相结合，可实现端到端的模型转换，也就是从训练端到生产端的一键式部署。

### MMDeploy 模型（MMDeploy Model）

也称 SDK Model。它是模型转换结果的集合。不仅包括后端模型，还包括模型的元信息。这些信息将用于推理 SDK 中。

### 推理 SDK（Inference SDK）

封装了模型的前处理、网络推理和后处理过程。对外提供多语言的模型推理接口。

## 准备工作

对于端到端的模型转换和推理，MMDeploy 依赖 Python 3.6+ 以及 PyTorch 1.8+。

**第一步**：从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda

**第二步**：创建并激活 conda 环境

```shell
conda create --name mmdeploy python=3.8 -y
conda activate mmdeploy
```

**第三步**: 参考[官方文档](https://pytorch.org/get-started/locally/)并安装 PyTorch

在 GPU 环境下：

```shell
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cudatoolkit={cudatoolkit_version} -c pytorch -c conda-forge
```

在 CPU 环境下：

```shell
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cpuonly -c pytorch
```

```{note}
在 GPU 环境下，请务必保证 {cudatoolkit_version} 和主机的 CUDA Toolkit 版本一致，避免在使用 TensorRT 时，可能引起的版本冲突问题。
```

## 安装 MMDeploy

**第一步**：通过 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmcv-full
```

**第二步**: 安装 MMDeploy 和 推理引擎

我们推荐用户使用预编译包安装和体验 MMDeploy 功能。请根据目标软硬件平台，从[这里](https://github.com/open-mmlab/mmdeploy/releases) 选择最新版本下载并安装。

目前，MMDeploy 的预编译包支持的平台和设备矩阵如下：

<table>
<thead>
  <tr>
    <th>OS-Arch</th>
    <th>Device</th>
    <th>ONNX Runtime</th>
    <th>TensorRT</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Linux-x86_64</td>
    <td>CPU</td>
    <td>Y</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>N</td>
    <td>Y</td>
  </tr>
  <tr>
    <td rowspan="2">Windows-x86_64</td>
    <td>CPU</td>
    <td>Y</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>N</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**注：对于不在上述表格中的软硬件平台，请参考[源码安装文档](01-how-to-build/build_from_source.md)，正确安装和配置 MMDeploy。**

以最新的预编译包为例，你可以参考以下命令安装：

<details open>
<summary><b>Linux-x86_64, CPU, ONNX Runtime 1.8.1</b></summary>

```shell
# 安装 MMDeploy ONNX Runtime 自定义算子库和推理 SDK
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.9.0/mmdeploy-0.9.0-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-0.9.0-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-0.9.0-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-0.9.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.9.0-cp38-none-linux_x86_64.whl
cd ..
# 安装推理引擎 ONNX Runtime
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

</details>

<details open>
<summary><b>Linux-x86_64, CUDA 11.x, TensorRT 8.2.3.0</b></summary>

```shell
# 安装 MMDeploy TensorRT 自定义算子库和推理 SDK
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.9.0/mmdeploy-0.9.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
tar -zxvf mmdeploy-0.9.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
cd mmdeploy-0.9.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
pip install dist/mmdeploy-0.9.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.9.0-cp38-none-linux_x86_64.whl
cd ..
# 安装推理引擎 TensorRT
# !!! 从 NVIDIA 官网下载 TensorRT-8.2.3.0 CUDA 11.x 安装包并解压到当前目录
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
# !!! 从 NVIDIA 官网下载 cuDNN 8.2.1 CUDA 11.x 安装包并解压到当前目录
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

</details>

<details open>
<summary><b>Windows-x86_64</b></summary>
</details>

请阅读 [这里](02-how-to-run/prebuilt_package_windows.md)，了解 MMDeploy 预编译包在 Windows 平台下的使用方法。

## 模型转换

在准备工作就绪后，我们可以使用 MMDeploy 中的工具 `tools/deploy.py`，将 OpenMMLab 的 PyTorch 模型转换成推理后端支持的格式。
对于`tools/deploy.py` 的使用细节，请参考 [如何转换模型](02-how-to-run/convert_model.md)。

以 [MMDetection](https://github.com/open-mmlab/mmdetection) 中的 `Faster R-CNN` 为例，我们可以使用如下命令，将 PyTorch 模型转换为 TenorRT 模型，从而部署到 NVIDIA GPU 上.

```shell
# 克隆 mmdeploy 仓库。转换时，需要使用 mmdeploy 仓库中的配置文件，建立转换流水线
git clone --recursive https://github.com/open-mmlab/mmdeploy.git

# 安装 mmdetection。转换时，需要使用 mmdetection 仓库中的模型配置文件，构建 PyTorch nn module
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

# 下载 Faster R-CNN 模型权重
wget -P checkpoints https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# 执行转换命令，实现端到端的转换
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    mmdetection/demo/demo.jpg \
    --work-dir mmdeploy_model/faster-rcnn \
    --device cuda \
    --dump-info
```

转换结果被保存在 `--work-dir` 指向的文件夹中。**该文件夹中不仅包含推理后端模型，还包括推理元信息。这些内容的整体被定义为 SDK Model。推理 SDK 将用它进行模型推理。**

```{tip}
在安装了 MMDeploy-ONNXRuntime 预编译包后，把上述转换命令中的detection_tensorrt_dynamic-320x320-1344x1344.py 换成 detection_onnxruntime_dynamic.py，并修改 --device 为 cpu，
即可以转出 onnx 模型，并用 ONNXRuntime 进行推理
```

## 模型推理

在转换完成后，你既可以使用 Model Converter 进行推理，也可以使用 Inference SDK。

### 使用 Model Converter 的推理 API

Model Converter 屏蔽了推理后端接口的差异，对其推理 API 进行了统一封装，接口名称为 `inference_model`。

以上文中 Faster R-CNN 的 TensorRT 模型为例，你可以使用如下方式进行模型推理工作：

```python
from mmdeploy.apis import inference_model
result = inference_model(
  model_cfg='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
  deploy_cfg='mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py',
  backend_files=['mmdeploy_model/faster-rcnn/end2end.engine'],
  img='mmdetection/demo/demo.jpg',
  device='cuda:0')
```

```{note}
接口中的 model_path 指的是推理引擎文件的路径，比如例子当中end2end.engine文件的路径。路径必须放在 list 中，因为有的推理引擎模型结构和权重是分开存储的。
```

### 使用推理 SDK

你可以直接运行预编译包中的 demo 程序，输入 SDK Model 和图像，进行推理，并查看推理结果。

```shell
cd mmdeploy-0.9.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
# 运行 python demo
python sdk/example/python/object_detection.py cuda ../mmdeploy_model/faster-rcnn ../mmdetection/demo/demo.jpg
# 运行 C/C++ demo
export LD_LIBRARY_PATH=$(pwd)/sdk/lib:$LD_LIBRARY_PATH
./sdk/bin/object_detection cuda ../mmdeploy_model/faster-rcnn ../mmdetection/demo/demo.jpg
```

```{note}
以上述命令中，输入模型是 SDK Model 的路径（也就是 Model Converter 中 --work-dir 参数），而不是推理引擎文件的路径。
因为 SDK 不仅要获取推理引擎文件，还需要推理元信息（deploy.json, pipeline.json）。它们合在一起，构成 SDK Model，存储在 --work-dir 下
```

除了 demo 程序，预编译包还提供了 SDK 多语言接口。你可以根据自己的项目需求，选择合适的语言接口，
把 MMDeploy SDK 集成到自己的项目中，进行二次开发。

#### Python API

对于检测功能，你也可以参考如下代码，集成 MMDeploy SDK Python API 到自己的项目中：

```python
from mmdeploy_python import Detector
import cv2

# 读取图片
img = cv2.imread('mmdetection/demo/demo.jpg')

# 创建检测器
detector = Detector(model_path='mmdeploy_models/faster-rcnn', device_name='cuda', device_id=0)
# 执行推理
bboxes, labels, _ = detector(img)
# 使用阈值过滤推理结果，并绘制到原图中
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```

更多示例，请查阅[这里](https://github.com/open-mmlab/mmdeploy/tree/master/demo/python)。

#### C++ API

使用 C++ API 进行模型推理的流程符合下面的模式：
![image](https://user-images.githubusercontent.com/4560679/182554486-2bf0ff80-9e82-4a0f-bccc-5e1860444302.png)

以下是具体过程：

```C++
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "mmdeploy/detector.hpp"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;

  // mmdeploy SDK model，以上文中转出的 faster r-cnn 模型为例
  std::string model_path = "mmdeploy_model/faster-rcnn";
  std::string image_path = "mmdetection/demo/demo.jpg";

  // 1. 读取模型
  mmdeploy::Model model(model_path);
  // 2. 创建预测器
  mmdeploy::Detector detector(model, mmdeploy::Device{device_name, device_id});
  // 3. 读取图像
  cv::Mat img = cv::imread(image_path);
  // 4. 应用预测器推理
  auto dets = detector.Apply(img);
  // 5. 处理推理结果: 此处我们选择可视化推理结果
  for (int i = 0; i < dets.size(); ++i) {
    const auto& box = dets[i].bbox;
    fprintf(stdout, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, dets[i].label_id, dets[i].score);
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }
  cv::imwrite("output_detection.png", img);
  return 0;
}
```

在您的项目CMakeLists中，增加：

```Makefile
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

编译时，使用 -DMMDeploy_DIR，传入MMDeloyConfig.cmake所在的路径。它在预编译包中的sdk/lib/cmake/MMDeloy下。
更多示例，请查阅[此处](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc/cpp)。

对于 C API、C# API、Java API 的使用方法，请分别阅读代码[C demos](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc/c)， [C# demos](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csharp) 和 [Java demos](https://github.com/open-mmlab/mmdeploy/tree/master/demo/java)。
我们将在后续版本中详细讲述它们的用法。

#### 加速预处理（实验性功能）

若要对预处理进行加速，请查阅[此处](./02-how-to-run/fuse_transform.md)

## 模型精度评估

为了测试部署模型的精度，推理效率，我们提供了 `tools/test.py` 来帮助完成相关工作。以上文中的部署模型为例：

```bash
python mmdeploy/tools/test.py \
    mmdeploy/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model mmdeploy_model/faster-rcnn/end2end.engine \
    --metrics ${METRICS} \
    --device cuda:0
```

```{note}
关于 --model 选项，当使用 Model Converter 进行推理时，它代表转换后的推理后端模型的文件路径。而当使用 SDK 测试模型精度时，该选项表示 MMDeploy Model 的路径.
```

请阅读 [如何进行模型评估](02-how-to-run/profile_model.md) 了解关于 `tools/test.py` 的使用细节。

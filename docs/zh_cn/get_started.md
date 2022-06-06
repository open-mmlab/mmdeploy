## 操作概述

MMDeploy提供了一系列工具，帮助您更轻松的将OpenMMLab下的算法部署到各种设备与平台上。您可以使用我们设计的流程一“部”到位，也可以定制您自己的转换流程。这份指引将会向您展示MMDeploy的基本使用方式，并帮助您将 MMDeploy SDK 整合进您的应用。

### 编译安装

首先我们需要根据[安装指南](./01-how-to-build/build_from_source.md)正确安装MMDeploy。**注意！** 不同推理后端的安装方式略有不同。可以根据下面的介绍选择最适合您的推理后端：

- [ONNXRuntime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html): ONNX Runtime 是一个跨平台的机器学习训练推理加速器，通过图形优化和变换以及硬件加速器提供优秀的推理性能。<span style="color:red">拥有完善的对ONNX的支持</span>。
- [TensorRT](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html): NVIDIA® TensorRT™ 是一个用于高性能深度学习推理的开发工具包（SDK）。借助Nvidia的设备特性，TensorRT可以优化模型的推理，提供更低的推理延迟以及更高的吞吐量。如果您希望将模型部署在<span style="color:red">NVIDIA硬件设备</span>上，那么TensorRT就是一个合适的选择。
- [ncnn](https://mmdeploy.readthedocs.io/en/latest/backends/ncnn.html): ncnn 是一个<span style="color:red">为手机端极致优化</span>的高性能神经网络前向计算框架。ncnn 从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台。基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能 APP，将 AI 带到您的指尖。
- [PPLNN](https://mmdeploy.readthedocs.io/en/latest/backends/pplnn.html): PPLNN是一个为高效AI推理所开发的高性能深度学习推理引擎。可以用于各种ONNX模型的推理。并且<span style="color:red">对OpenMMLab有非常强的支持</span>。
- [OpenVINO](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html): OpenVINO™ 是一个为优化与部署AI推理开发的开源工具集。该工具集<span style="color:red">可无缝集成到 Intel 硬件平台</span>，包括最新的神经网络加速芯片，Intel计算棒，边缘设备等。

选择最适合您的推理后端，点击对应的连接查看具体安装细节

### 模型转换

一旦您完成了MMDeploy的安装，就可以用一条指令轻松的将OpenMMLab的PyTorch模型转换成推理后端支持的格式！以 [MMDetection](https://github.com/open-mmlab/mmdetection) 中的 `Faster-RCNN` 到 `TensorRT` 的转换为例：

```bash
# 本例假设 MMDeploy 所在目录为 ${MMDEPLOY_DIR}， MMDetection 所在目录为 ${MMDET_DIR}
# 如果您不知道具体的安装位置，可以在终端通过命令 `pip show mmdeploy` 和 `pip show mmdet` 查看

python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` 是一个方便模型转换的工具。可以阅读 [如何转换模型](./02-how-to-run/convert_model.md) 了解更多细节。转换后的模型以及一些其他的信息将会被保存在 `${WORK_DIR}` 中。MMDeploy SDK 可以使用这些信息进行模型推理。

`detection_tensorrt_dynamic-320x320-1344x1344.py` 是一个包含所有转换需要的可配置参数的配置文件。该文件的命名遵循如下规则：

```bash
<任务名>_<推理后端>-[后端特性]_<动态模型支持>.py
```

可以很容易的通过文件名来确定最适合的那个配置文件。如果您希望定制自己的转换配置，可以修改配置文件中的具体条目。我们提供了 [如何编写配置文件](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_write_config.html) 来指导您如何进行编辑。

### 模型推理

得到了转换后的模型之后，就可以使用推理后端提供的API来进行推理。也许您想绕过API的学习与开发，确认下转换后的模型效果。我们提供了对这些API的统一封装：

```python
from mmdeploy.apis import inference_model

result = inference_model(model_cfg, deploy_cfg, backend_files, img=img, device=device)
```

`inference_model`会创建一个对后端模型的封装，通过该封装进行推理。推理的结果会保持与OpenMMLab中原模型同样的格式。

### 模型评估

转换后的模型是否会带来一些精度损失？推理后端是否为模型带来效率提升？我们提供了工具来帮助完成验证与评估工作。以TensorRT的Faster-RCNN模型为例：

```bash
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

请阅读 [如何进行模型评估](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_evaluate_a_model.html) 了解关于 `tools/test.py` 的使用细节。

### 整合 MMDeploy SDK

请参考[安装指南](./01-how-to-build/build_from_source.md)，在编译 MMDeploy 时开启`MMDEPLOY_BUILD_SDK`以安装 MMDeploy SDK。
成功安装后，安装目录的文件结构将会如下所示：

```
install
├── example
├── include
│   ├── c
│   └── cpp
└── lib
```

其中 `include/c` 和 `include/cpp` 分别对应 C 与 C++ API。

**注意：C++的API尚处于不稳定阶段，可能会在未来发生变动，暂时不推荐使用**

在 example 目录下有包含分类，目标检测，图像分割等数个范例。可以通过他们学习如何使用 MMDeploy SDK 以及如何将 ${MMDeploy_LIBS} 链接到应用程序。

### 从零开始进行模型部署

这章节将会展示如何从零开始进行 MMDetection 中 Faster-RCNN 的模型部署与推理。

#### 创建虚拟环境并安装 MMDetection

请运行下面的指令在Anaconda环境中[安装MMDetection](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html#a-from-scratch-setup-script)。

```bash
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch -y

# 安装 mmcv
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8/index.html

# 安装mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

#### 下载 Faster R-CNN 的模型文件

请从[本链接](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)下载模型文件，放在`{MMDET_ROOT}/checkpoints`目录下。其中`{MMDET_ROOT}`为您的MMDetection的根目录。

#### 安装 MMDeploy 以及 ONNX Runtime

请运行下面的指令在Anaconda环境中[安装MMDeploy](./01-how-to-build/build_from_source.md)。

```bash
conda activate openmmlab

git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
pip install -e .
```

一旦我们完成MMDeploy的安装，我们需要选择一个模型的推理引擎。这里我们以ONNX Runtime为例。运行下面命令来[安装ONNX Runtime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)：

```bash
pip install onnxruntime==1.8.1
```

然后下载 ONNX Runtime Library来编译 MMDeploy 中的算子插件：

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

cd ${MMDEPLOY_DIR} # 到 MMDeploy 根目录下
mkdir -p build && cd build

# 编译自定义算子
cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc)

# 编译 MMDeploy SDK
cmake -DMMDEPLOY_BUILD_SDK=ON \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DOpenCV_DIR=/path/to/OpenCV/lib/cmake/OpenCV \
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=mmdet ..
make -j$(nproc) && make install
```

#### 模型转换

当我们成功安装 MMDetection, MMDeploy, ONNX Runtime 以及编译ONNX Runtime的插件之后，我们就可以将 Faster-RCNN 转换成 ONNX 格式，以支持 ONNX Runtime 下的模型推理。运行下面指令来使用我们的部署工具：

```bash
# 本例假设 MMDeploy 所在目录为 ${MMDEPLOY_DIR}， MMDetection 所在目录为 ${MMDET_DIR}
# 如果您不知道具体的安装位置，可以在终端通过命令 `pip show mmdeploy` 和 `pip show mmdet` 查看

python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${MMDET_DIR}/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${MMDET_DIR}/demo/demo.jpg \
    --work-dir work_dirs \
    --device cpu \
    --show \
    --dump-info
```

如果脚本运行成功，屏幕上将先后显示两张图片。首先是ONNX Runtime推理结果的可视化，然后是PyTorch推理结果的可视化。同时，ONNX文件 `end2end.onnx` 以及数个json文件（供SDK使用）将会被生成在`work_dirs`目录下。

#### 运行 MMDeploy SDK demo

进行模型转换后，SDK模型将被保存在`work_dirs`目录下。
可以用如下方法编译并运行 MMDeploy SDK demo。

```bash
cd build/install/example

# 配置ONNX Runtime库目录
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib
mkdir -p build && cd build
cmake -DOpenCV_DIR=path/to/OpenCV/lib/cmake/OpenCV \
      -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy ..
make object_detection
# 设置log等级
export SPDLOG_LEVEL=warn
# 运行目标检测范例程序
./object_detection cpu ${work_dirs} ${path/to/an/image}
```

如果demo运行成功，将会生成一张名为"output_detection.png"的图片，展示模型输出的可视化效果。

### 新模型的支持？

如果您希望使用的模型尚未被 MMDeploy 所支持，您可以尝试自己添加对应的支持。我们准备了如下的文档：
- 请阅读[如何支持新模型](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_support_new_models.html)了解我们的模型重写机制。

最后，欢迎大家踊跃提PR。

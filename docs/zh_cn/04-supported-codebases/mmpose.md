# MMPose 模型部署

- [MMPose 模型部署](#mmpose-模型部署)
  - [安装](#安装)
    - [安装 mmpose](#安装-mmpose)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
  - [模型支持列表](#模型支持列表)

______________________________________________________________________

[MMPose](https://github.com/open-mmlab/mmpose/tree/main)，又称 `mmpose`，是一个基于 PyTorch 的姿态估计的开源工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

## 安装

### 安装 mmpose

请参考[官网安装指南](https://mmpose.readthedocs.io/en/latest/installation.html#best-practices)。

### 安装 mmdeploy

mmdeploy 有以下几种安装方式:

**方式一：** 安装预编译包

请参考[安装概述](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html#mmdeploy)

**方式二：** 一键式脚本安装

如果部署平台是 **Ubuntu 18.04 及以上版本**， 请参考[脚本安装说明](../01-how-to-build/build_from_script.md)，完成安装过程。
比如，以下命令可以安装 mmdeploy 以及配套的推理引擎——`ONNX Runtime`.

```shell
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**方式三：** 源码安装

在方式一、二都满足不了的情况下，请参考[源码安装说明](../01-how-to-build/build_from_source.md) 安装 mmdeploy 以及所需推理引擎。

## 模型转换

你可以使用 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) 把 mmpose 模型一键式转换为推理后端模型。
该工具的详细使用说明请参考[这里](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

以下，我们将演示如何把 `hrnet` 转换为 onnx 模型。

```shell
cd mmdeploy
# download hrnet model from mmpose model zoo
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest .
# convert mmdet model to onnxruntime model with static shape
python tools/deploy.py \
    configs/mmpose/pose-detection_onnxruntime_static.py \
    td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmpose/ort \
    --device cpu \
    --show
```

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpose)。
文件的命名模式是：

```
pose-detection_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等
- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32
- **{static | dynamic}:** 动态、静态 shape
- **{shape}:** 模型输入的 shape 或者 shape 范围

在上例中，你也可以把 `hrnet` 转为其他后端模型。比如使用`pose-detection_tensorrt_static-256x192.py`，把模型转为 tensorrt 模型。

```{tip}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmpose/ort`，结构如下：

```
mmdeploy_models/mmpose/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

重要的是：

- **end2end.onnx**: 推理引擎文件。可用 ONNX Runtime 推理
- \***.json**:  mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为**mmdeploy SDK model**。换言之，**mmdeploy SDK model**既包括推理引擎，也包括推理 meta 信息。

## 模型推理

### 后端模型推理

以上述模型转换后的 `end2end.onnx` 为例，你可以使用如下代码进行推理：

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmpose/pose-detection_onnxruntime_static.py'
model_cfg = 'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmpose/ort/end2end.onnx']
image = './demo/resources/human-pose.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_pose.png')
```

### SDK 模型推理

> TODO

## 模型支持列表

| Model                                                                                                     | Task          | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :-------------------------------------------------------------------------------------------------------- | :------------ | :----------: | :------: | :--: | :---: | :------: |
| [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019)          | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [MSPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019)           | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [LiteHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021)  | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [Hourglass](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#hourglass-eccv-2016) | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [SimCC](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022)         | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)                                | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [YoloX-Pose](https://github.com/open-mmlab/mmpose/tree/main/projects/yolox_pose)                          | PoseDetection |      Y       |    Y     |  N   |   N   |    Y     |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmo)                                   | PoseDetection |      Y       |    Y     |  N   |   N   |    N     |

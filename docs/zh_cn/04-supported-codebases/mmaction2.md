# MMAction2 模型部署

- [MMAction2 模型部署](#mmaction2-模型部署)
  - [安装](#安装)
    - [安装 mmaction2](#安装-mmaction2)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
    - [视频分类任务模型转换](#视频分类任务模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
      - [视频分类 SDK 模型推理](#视频分类-sdk-模型推理)
  - [模型支持列表](#模型支持列表)

______________________________________________________________________

[MMAction2](https://github.com/open-mmlab/mmaction2)是一款基于 PyTorch 的视频理解开源工具箱，是[OpenMMLab](https://openmmlab.com)项目的成员之一。

## 安装

### 安装 mmaction2

请参考[官网安装指南](https://github.com/open-mmlab/mmaction2#installation).

### 安装 mmdeploy

mmdeploy 有以下几种安装方式:

**方式一：** 安装预编译包

通过此[链接](https://github.com/open-mmlab/mmdeploy/releases)获取最新的预编译包

**方式二：** 一键式脚本安装

如果部署平台是 **Ubuntu 18.04 及以上版本**， 请参考[脚本安装说明](../01-how-to-build/build_from_script.md)，完成安装过程。
比如，以下命令可以安装 mmdeploy 以及配套的推理引擎——`ONNX Runtime`.

```shell
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**方式三：** 源码安装

在方式一、二都满足不了的情况下，请参考[源码安装说明](../01-how-to-build/build_from_source.md) 安装 mmdeploy 以及所需推理引擎。

## 模型转换

你可以使用 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py) 把 mmaction2 模型一键式转换为推理后端模型。
该工具的详细使用说明请参考[这里](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md#usage).

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/master/configs/mmaction)。
文件的命名模式是：

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{task}:** mmaction2 中的任务
- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等
- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32
- **{static | dynamic}:** 动态、静态 shape
- **{shape}:** 模型输入的 shape 或者 shape 范围
- **{2d/3d}:** 表示模型的类别

以下，我们将演示如何把视频分类任务中 `tsn` 模型转换为 onnx 模型。

### 视频分类任务模型转换

```shell
cd mmdeploy

# download tsn model from mmaction2 model zoo
mim download mmaction2 --config tsn_r50_1x1x3_100e_kinetics400_rgb --dest .

# convert mmaction2 model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py \
    tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth \
    tests/data/arm_wrestling.mp4 \
    --work-dir mmdeploy_models/mmaction/tsn/ort \
    --device cpu \
    --show \
    --dump-info
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmaction/tsn/ort`，结构如下：

```
mmdeploy_models/mmaction/tsn/ort
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
import numpy as np
import torch

deploy_cfg = 'configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py'
model_cfg = 'tsn_r50_1x1x3_100e_kinetics400_rgb.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmaction/tsn/ort/end2end.onnx']
image = 'tests/data/arm_wrestling.mp4'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.init_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = task_processor.run_inference(model, model_inputs)

# show top5-results
result = np.array(result[0])
top_index = np.argsort(result)[::-1]
for i in range(5):
    index = top_index[i]
    print(index, result[index])
```

### SDK 模型推理

你也可以参考如下代码，对 SDK model 进行推理：

#### 视频分类 SDK 模型推理

```python
from mmdeploy_python import VideoRecognizer
import cv2

# refer to demo/python/video_recognition.py
# def SampleFrames(cap, clip_len, frame_interval, num_clips):
#  ...

cap = cv2.VideoCapture('tests/data/arm_wrestling.mp4')

clips, info = SampleFrames(cap, 1, 1, 25)

# create a recognizer
recognizer = VideoRecognizer(model_path='./mmdeploy_models/mmaction/tsn/ort', device_name='cpu', device_id=0)
# perform inference
result = recognizer(clips, info)
# show inference result
for label_id, score in result:
    print(label_id, score)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/master/demo)学习其他语言接口的使用方法。

> mmaction2 的 C#，Java接口待开发

## 模型支持列表

| Model                                                                                        | TorchScript | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :------------------------------------------------------------------------------------------- | :---------: | :----------: | :------: | :--: | :---: | :------: |
| [TSN](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn)           |      N      |      Y       |    Y     |  N   |   N   |    N     |
| [SlowFast](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast) |      N      |      Y       |    Y     |  N   |   N   |    N     |

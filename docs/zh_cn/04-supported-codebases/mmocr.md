# MMOCR 模型部署

- [MMOCR 模型部署](#mmocr-模型部署)
  - [安装](#安装)
    - [安装 mmocr](#安装-mmocr)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
    - [文字检测任务模型转换](#文字检测任务模型转换)
    - [文字识别任务模型转换](#文字识别任务模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
      - [文字检测 SDK 模型推理](#文字检测-sdk-模型推理)
      - [文字识别 SDK 模型推理](#文字识别-sdk-模型推理)
  - [模型支持列表](#模型支持列表)
  - [注意事项](#注意事项)

______________________________________________________________________

[MMOCR](https://github.com/open-mmlab/mmocr/tree/main)，又称 `mmocr`，是基于 PyTorch 和 mmdetection 的开源工具箱，专注于文本检测，文本识别以及相应的下游任务，如关键信息提取。 它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装

### 安装 mmocr

请参考[官网安装指南](https://mmocr.readthedocs.io/en/latest/get_started/install.html).

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

你可以使用 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) 把 mmocr 模型一键式转换为推理后端模型。
该工具的详细使用说明请参考[这里](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmocr)。
文件的命名模式是：

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{task}:** mmocr 中的任务

  mmdeploy 支持 mmocr 中的文字检测（text detection）、文字识别（text recognition）任务中的模型。关于`模型-任务`的划分，请参考章节[模型支持列表](#模型支持列表)。

  **请务必**使用对应的部署文件转换相关的模型。

- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等

- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32

- **{static | dynamic}:** 动态、静态 shape

- **{shape}:** 模型输入的 shape 或者 shape 范围

在接下来来的两个章节，我们将分别演示文字检测任务中的`dbnet`模型，和文字识别任务中的`crnn`模型转换 onnx 模型的方法。

### 文字检测任务模型转换

```shell
cd mmdeploy
# download dbnet model from mmocr model zoo
mim download mmocr --config dbnet_resnet18_fpnc_1200e_icdar2015 --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py \
    dbnet_resnet18_fpnc_1200e_icdar2015.py \
    dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth \
    demo/resources/text_det.jpg \
    --work-dir mmdeploy_models/mmocr/dbnet/ort \
    --device cpu \
    --show \
    --dump-info
```

### 文字识别任务模型转换

```shell
cd mmdeploy
# download crnn model from mmocr model zoo
mim download mmocr --config crnn_mini-vgg_5e_mj --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py \
    crnn_mini-vgg_5e_mj.py \
    crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth \
    demo/resources/text_recog.jpg \
    --work-dir mmdeploy_models/mmocr/crnn/ort \
    --device cpu \
    --show \
    --dump-info
```

你也可以把它们转为其他后端模型。比如使用`text-detection/text-detection_tensorrt-_dynamic-320x320-2240x2240.py`，把 `dbnet` 模型转为 tensorrt 模型。

```{tip}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

[文字检测任务模型转换](#文字检测任务模型转换)例子中的`mmdeploy_models/mmocr/dbnet/ort`，结构如下：

```
mmdeploy_models/mmocr/dbnet/ort
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

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py'
model_cfg = 'dbnet_resnet18_fpnc_1200e_icdar2015.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmocr/dbnet/ort/end2end.onnx']
image = './demo/resources/text_det.jpg'

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
    output_file='output_ocr.png')
```

**提示：**

在这个脚本中，把'deploy_cfg', 'model_cfg', 'backend_model' and 'image' 替换为[文字识别任务模型转换](#文字识别任务模型转换)中对应的参数，
就可以推理 `crnn` onnx 模型了。

### SDK 模型推理

#### 文字检测 SDK 模型推理

你也可以参考如下代码，对 `dbnet` SDK model 进行推理：

```python
import cv2
from mmdeploy_runtime import TextDetector

img = cv2.imread('demo/resources/text_det.jpg')
# create text detector
detector = TextDetector(
    model_path='mmdeploy_models/mmocr/dbnet/ort',
    device_name='cpu',
    device_id=0)
# do model inference
bboxes = detector(img)
# draw detected bbox into the input image
if len(bboxes) > 0:
    pts = ((bboxes[:, 0:8] + 0.5).reshape(len(bboxes), -1,
                                          2).astype(int))
    cv2.polylines(img, pts, True, (0, 255, 0), 2)
    cv2.imwrite('output_ocr.png', img)
```

#### 文字识别 SDK 模型推理

```python
import cv2
from mmdeploy_runtime import TextRecognizer

img = cv2.imread('demo/resources/text_recog.jpg')
# create text recognizer
recognizer = TextRecognizer(
  model_path='mmdeploy_models/mmocr/crnn/ort',
  device_name='cpu',
  device_id=0
)
# do model inference
texts = recognizer(img)
# print the result
print(texts)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 模型支持列表

| Model                                                                                | Task             | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |
| :----------------------------------------------------------------------------------- | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: |
| [DBNet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet)         | text-detection   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DBNetpp](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnetpp)     | text-detection   |      N      |      Y      |    Y     |  ?   |   ?   |    Y     |
| [PSENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet)       | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet)         | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [TextSnake](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/textsnake) | text-detection   |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/maskrcnn)   | text-detection   |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |
| [CRNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn)         | text-recognition |      Y      |      Y      |    Y     |  Y   |   Y   |    N     |
| [SAR](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/sar)           | text-recognition |      N      |      Y      |    Y     |  N   |   N   |    N     |
| [SATRN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/satrn)       | text-recognition |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [ABINet](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet)     | text-recognition |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |

## 注意事项

- ABINet 在 TensorRT 后端要求使用 pytorch1.10+， TensorRT 8.4+。

- SAR 在网络推广中使用 `valid_ratio`，这会让导出的 ONNX 文件精度下降。当测试图片的 `valid_ratio`s 和转换图片的值差异很大，这种下降就会越多。

- 对于 TensorRT 后端，用户需要使用正确的配置文件。比如 CRNN 只接受单通道输入。下面是一个示例表格:

  | Model    | Config                                                     |
  | :------- | :--------------------------------------------------------- |
  | MaskRCNN | text-detection_mrcnn_tensorrt_dynamic-320x320-2240x2240.py |
  | CRNN     | text-recognition_tensorrt_dynamic-1x32x32-1x32x640.py      |
  | SATRN    | text-recognition_tensorrt_dynamic-32x32-32x640.py          |
  | SAR      | text-recognition_tensorrt_dynamic-48x64-48x640.py          |
  | ABINet   | text-recognition_tensorrt_static-32x128.py                 |

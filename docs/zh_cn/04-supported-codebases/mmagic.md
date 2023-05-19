# MMagic 模型部署

- [MMagic 模型部署](#mmagic-模型部署)
  - [安装](#安装)
    - [安装 mmagic](#安装-mmagic)
    - [安装 mmdeploy](#安装-mmdeploy)
  - [模型转换](#模型转换)
    - [超分任务模型转换](#超分任务模型转换)
  - [模型规范](#模型规范)
  - [模型推理](#模型推理)
    - [后端模型推理](#后端模型推理)
    - [SDK 模型推理](#sdk-模型推理)
  - [模型支持列表](#模型支持列表)

______________________________________________________________________

[MMagic](https://github.com/open-mmlab/mmagic)，又称 `mmagic`，是基于 PyTorch 的开源图像和视频编辑工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目成员之一。

## 安装

### 安装 mmagic

请参考[官网安装指南](https://github.com/open-mmlab/mmagic/tree/main#installation)。

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

你可以使用 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) 把 mmagic 模型一键式转换为推理后端模型。
该工具的详细使用说明请参考[这里](https://github.com/open-mmlab/mmdeploy/tree/main/docs/zh_cn/02-how-to-run/convert_model.md#使用方法).

转换的关键之一是使用正确的配置文件。项目中已内置了各后端部署[配置文件](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmagic)。
文件的命名模式是：

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

其中：

- **{task}:** mmagic 中的任务

  mmagic 中任务有多种。目前，mmdeploy 支持其中的超分（super resolution）任务。关于`模型-任务`的划分，请参考章节[模型支持列表](#模型支持列表)。

  **请务必**使用对应的部署文件转换相关的模型。

- **{backend}:** 推理后端名称。比如，onnxruntime、tensorrt、pplnn、ncnn、openvino、coreml 等等

- **{precision}:** 推理精度。比如，fp16、int8。不填表示 fp32

- **{static | dynamic}:** 动态、静态 shape

- **{shape}:** 模型输入的 shape 或者 shape 范围

### 超分任务模型转换

以下，我们将演示如何把 `ESRGAN` 转换为 onnx 模型。

```shell
cd mmdeploy
# download esrgan model from mmagic model zoo
mim download mmagic --config esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k --dest .
# convert esrgan model to onnxruntime model with dynamic shape
python tools/deploy.py \
  configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py \
  esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py \
  esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth \
  demo/resources/face.png \
  --work-dir mmdeploy_models/mmagic/ort \
  --device cpu \
  --show \
  --dump-info
```

你也可以把 `ESRGAN` 转为其他后端模型。比如使用`super-resolution/super-resolution_tensorrt-_dynamic-32x32-512x512.py`，把模型转为 tensorrt 模型。

```{tip}
当转 tensorrt 模型时, --device 需要被设置为 "cuda"
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmagic/ort`，结构如下：

```
mmdeploy_models/mmagic/ort
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

deploy_cfg = 'configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py'
model_cfg = 'esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmagic/ort/end2end.onnx']
image = './demo/resources/face.png'

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
    output_file='output_restorer.bmp')
```

### SDK 模型推理

你也可以参考如下代码，对 SDK model 进行推理：

```python
from mmdeploy_runtime import Restorer
import cv2

img = cv2.imread('./demo/resources/face.png')

# create a classifier
restorer = Restorer(model_path='./mmdeploy_models/mmagic/ort', device_name='cpu', device_id=0)
# perform inference
result = restorer(img)

# visualize inference result
# convert to BGR
result = result[..., ::-1]
cv2.imwrite('output_restorer.bmp', result)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 模型支持列表

| Model                                                                             | Task             | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :-------------------------------------------------------------------------------- | :--------------- | :----------: | :------: | :--: | :---: | :------: |
| [SRCNN](https://github.com/open-mmlab/mmagic/tree/main/configs/srcnn)             | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/esrgan)           | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ESRGAN-PSNR](https://github.com/open-mmlab/mmagic/tree/main/configs/esrgan)      | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)      | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SRResNet](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)   | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [Real-ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/real_esrgan) | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [EDSR](https://github.com/open-mmlab/mmagic/tree/main/configs/edsr)               | super-resolution |      Y       |    Y     |  Y   |   N   |    Y     |
| [RDN](https://github.com/open-mmlab/mmagic/tree/main/configs/rdn)                 | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |

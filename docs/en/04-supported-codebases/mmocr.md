# MMOCR Deployment

- [MMOCR Deployment](#mmocr-deployment)
  - [Installation](#installation)
    - [Install mmocr](#install-mmocr)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
    - [Convert text detection model](#convert-text-detection-model)
    - [Convert text recognition model](#convert-text-recognition-model)
  - [Model Specification](#model-specification)
  - [Model Inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)
  - [Reminder](#reminder)

______________________________________________________________________

[MMOCR](https://github.com/open-mmlab/mmocr/tree/1.x) is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmocr

Please follow the [installation guide](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html) to install mmocr. If you have already done that, please move on to [the next section](#install-mmdeploy).

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

> **TODO**. MMDeploy hasn't released based on dev-1.x branch.

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md). For example, the following commands install mmdeploy as well as inference engine - `ONNX Runtime`.

```shell
git clone --recursive -b dev-1.x https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/tools/deploy.py) to convert mmocr models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md#usage).

When using `tools/deploy.py`, it is crucial to specify the correct deployment config. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmocr) of all supported backends for mmocr, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmocr.

  MMDeploy supports models of two tasks of mmocr, one is `text detection` and the other is `text-recogntion`.

  **DO remember to use** the corresponding deployment config file when trying to converting models of different tasks.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml and etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

In the next two chapters, we will task `dbnet` model from `text detection` task and `crnn` model from `text recognition` task respectively as examples, showing how to convert them to onnx model that can be inferred by ONNX Runtime.

### Convert text detection model

```shell
cd mmdeploy
# download dbnet model from mmocr model zoo
mim download mmocr --config dbnet_resnet18_fpnc_1200e_icdar2015 --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py \
    dbnet_resnet18_fpnc_1200e_icdar2015.py \
    dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth \
    resources/converter/text_det.jpg \
    --work-dir mmdeploy_models/mmocr/dbnet/ort \
    --device cpu \
    --show \
    --dump-info
```

### Convert text recognition model

```shell
cd mmdeploy
# download crnn model from mmocr model zoo
mim download mmocr --config crnn_mini-vgg_5e_mj --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py \
    crnn_mini-vgg_5e_mj.py \
    crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth \
    resources/converter/text_recog.jpg \
    --work-dir mmdeploy_models/mmocr/crnn/ort \
    --device cpu \
    --show \
    --dump-info
```

You can also convert the above models to other backend models by changing the deployment config file `*_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmocr), e.g., converting `dbnet` to tensorrt-fp32 model by `text-detection/text-detection_tensorrt-_dynamic-320x320-2240x2240.py`.

```{tip}
When converting mmocr models to tensorrt models, --device should be set to "cuda"
```

## Model Specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmocr/ort/text-detection` in the previous example. It includes:

```
mmdeploy_models/mmocr/dbnet/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- **deploy.json**: meta information about backend model
- **pipeline.json**: inference pipeline of mmdeploy SDK
- **detail.json**: conversion parameters

The whole package **mmdeploy_models/mmocr/dbnet/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model Inference

### Backend model inference

MMDeploy provides a unified API named as `inference_model` to inference model, making all inference backends API transparent to users.

Take the previous converted `end2end.onnx` mode of `dbnet` as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py'
model_cfg = 'dbnet_resnet18_fpnc_1200e_icdar2015.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmocr/dbnet/ort/end2end.onnx']
image = './resources/converter/text_det.jpg'

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

**Tip**:

Map 'deploy_cfg', 'model_cfg', 'backend_model' and 'image' to corresponding arguments in chapter [convert text recognition model](#convert-text-recognition-model), you will get the ONNX Runtime inference results of `crnn` onnx model.

### SDK model inference

TODO

## Supported models

| Model                                                                          | Task             | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |
| :----------------------------------------------------------------------------- | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: |
| [DBNet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet)   | text-detection   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [PSENet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/psenet) | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [PANet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/panet)   | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn)   | text-recognition |      Y      |      Y      |    Y     |  Y   |   Y   |    N     |
| [SAR](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)     | text-recognition |      N      |      Y      |    N     |  N   |   N   |    N     |
| [SATRN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/satrn) | text-recognition |      Y      |      Y      |    Y     |  N   |   N   |    N     |

## Reminder

Note that ncnn, pplnn, and OpenVINO only support the configs of DBNet18 for DBNet.

For the PANet with the [checkpoint](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth) pretrained on ICDAR dataset, if you want to convert the model to TensorRT with 16 bits float point, please try the following script.

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend

FACTOR = 32
ENABLE = False
CHANNEL_THRESH = 400


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.necks.FPEM_FFM.forward',
    backend=Backend.TENSORRT.value)
def fpem_ffm__forward__trt(ctx, self, x: Sequence[torch.Tensor], *args,
                           **kwargs) -> Sequence[torch.Tensor]:
    """Rewrite `forward` of FPEM_FFM for tensorrt backend.

    Rewrite this function avoid overflow for tensorrt-fp16 with the checkpoint
    `https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm
    _sbn_600e_icdar2015_20210219-42dbe46a.pth`

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class FPEM_FFM.
        x (List[Tensor]): A list of feature maps of shape (N, C, H, W).

    Returns:
        outs (List[Tensor]): A list of feature maps of shape (N, C, H, W).
    """
    c2, c3, c4, c5 = x
    # reduce channel
    c2 = self.reduce_conv_c2(c2)
    c3 = self.reduce_conv_c3(c3)
    c4 = self.reduce_conv_c4(c4)

    if ENABLE:
        bn_w = self.reduce_conv_c5[1].weight / torch.sqrt(
            self.reduce_conv_c5[1].running_var + self.reduce_conv_c5[1].eps)
        bn_b = self.reduce_conv_c5[
            1].bias - self.reduce_conv_c5[1].running_mean * bn_w
        bn_w = bn_w.reshape(1, -1, 1, 1).repeat(1, 1, c5.size(2), c5.size(3))
        bn_b = bn_b.reshape(1, -1, 1, 1).repeat(1, 1, c5.size(2), c5.size(3))
        conv_b = self.reduce_conv_c5[0].bias.reshape(1, -1, 1, 1).repeat(
            1, 1, c5.size(2), c5.size(3))
        c5 = FACTOR * (self.reduce_conv_c5[:-1](c5)) - (FACTOR - 1) * (
            bn_w * conv_b + bn_b)
        c5 = self.reduce_conv_c5[-1](c5)
    else:
        c5 = self.reduce_conv_c5(c5)

    # FPEM
    for i, fpem in enumerate(self.fpems):
        c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
        if i == 0:
            c2_ffm = c2
            c3_ffm = c3
            c4_ffm = c4
            c5_ffm = c5
        else:
            c2_ffm += c2
            c3_ffm += c3
            c4_ffm += c4
            c5_ffm += c5

    # FFM
    c5 = F.interpolate(
        c5_ffm,
        c2_ffm.size()[-2:],
        mode='bilinear',
        align_corners=self.align_corners)
    c4 = F.interpolate(
        c4_ffm,
        c2_ffm.size()[-2:],
        mode='bilinear',
        align_corners=self.align_corners)
    c3 = F.interpolate(
        c3_ffm,
        c2_ffm.size()[-2:],
        mode='bilinear',
        align_corners=self.align_corners)
    outs = [c2_ffm, c3, c4, c5]
    return tuple(outs)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.backbones.resnet.BasicBlock.forward',
    backend=Backend.TENSORRT.value)
def basic_block__forward__trt(ctx, self, x: torch.Tensor) -> torch.Tensor:
    """Rewrite `forward` of BasicBlock for tensorrt backend.

    Rewrite this function avoid overflow for tensorrt-fp16 with the checkpoint
    `https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm
    _sbn_600e_icdar2015_20210219-42dbe46a.pth`

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class FPEM_FFM.
        x (Tensor): The input tensor of shape (N, C, H, W).

    Returns:
        outs (Tensor): The output tensor of shape (N, C, H, W).
    """
    if self.conv1.in_channels < CHANNEL_THRESH:
        return ctx.origin_func(self, x)

    identity = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = self.relu(out)

    out = self.conv2(out)

    if torch.abs(self.norm2(out)).max() < 65504:
        out = self.norm2(out)
        out += identity
        out = self.relu(out)
        return out
    else:
        global ENABLE
        ENABLE = True
        # the output of the last bn layer exceeds the range of fp16
        w1 = self.norm2.weight / torch.sqrt(self.norm2.running_var +
                                            self.norm2.eps)
        bias = self.norm2.bias - self.norm2.running_mean * w1
        w1 = w1.reshape(1, -1, 1, 1).repeat(1, 1, out.size(2), out.size(3))
        bias = bias.reshape(1, -1, 1, 1).repeat(1, 1, out.size(2),
                                                out.size(3)) + identity
        out = self.relu(w1 * (out / FACTOR) + bias / FACTOR)

        return out

```

# mmocr 模型支持列表

mmocr 是一个基于 PyTorch 和 mmdetection 的开源工具箱，用于文本检测，文本识别以及相应的下游任务，例如关键信息提取，是 [OpenMMLab](https://openmmlab.com/)项目的一部分。

## 安装

参照 [install.md](https://mmocr.readthedocs.io/en/latest/install.html)。

## 支持列表

| Model  | Task             | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |                                  Model config                                   |
| :----- | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: | :-----------------------------------------------------------------------------: |
| DBNet  | text-detection   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |  [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet)  |
| PSENet | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/psenet)  |
| PANet  | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |  [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/panet)  |
| CRNN   | text-recognition |      Y      |      Y      |    Y     |  Y   |   Y   |    N     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn)  |
| SAR    | text-recognition |      N      |      Y      |    N     |  N   |   N   |    N     |  [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)  |
| SATRN  | text-recognition |      Y      |      Y      |    Y     |  N   |   N   |    N     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/satrn) |

## 注意事项

请注意，ncnn、pplnn 和 OpenVINO 仅支持 DBNet 的 DBNet18 配置。

对于在 ICDAR 数据集上预训 [checkpoint](https://download.openmmlab.com/mmocr/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth) 的 PANet，如果要将模型转为具有 fp16 TensorRT，请尝试以下脚本。

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
    func_name='mmdet.models.backbones.resnet.BasicBlock.forward',
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

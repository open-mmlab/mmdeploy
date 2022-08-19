# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.backbones.shufflenet_v2.InvertedResidual.forward')
def shufflenetv2_backbone__forward__default(ctx, self, x):
    """Rewrite `forward` of InvertedResidual used in shufflenet_v2.

    The chunk in original InvertedResidual.forward will convert to dynamic
    `Slice` operator in ONNX, which will raise error in ncnn, torchscript
     and tensorrt. Here we replace `chunk` with `split`.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (InvertedResidual): The instance of the class InvertedResidual.
        x (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from InvertedResidual. The tensor
        shape (N, Cout, H, W).
    """
    from mmcls.models.utils import channel_shuffle
    if self.stride > 1:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    else:
        assert x.shape[1] % 2 == 0
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)

    out = channel_shuffle(out, 2)

    return out

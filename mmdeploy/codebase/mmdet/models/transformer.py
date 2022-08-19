# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.utils.transformer.PatchMerging.forward',
    backend='tensorrt')
def patch_merging__forward__tensorrt(ctx, self, x, input_size):
    """Rewrite forward function of PatchMerging class for TensorRT.
    In original implementation, mmdet applies nn.unfold to accelerate the
    inferece. However, the onnx graph of it can not be parsed correctly by
    TensorRT. In mmdeploy, it is replaced.
    Args:
        x (Tensor): Has shape (B, H*W, C_in).
        input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
            Default: None.
    Returns:
        tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
            - out_size (tuple[int]): Spatial shape of x, arrange as
                (Merged_H, Merged_W).
    """
    H, W = input_size
    B, L, C = x.shape
    assert L == H * W, 'input feature has wrong size'
    assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'

    x = x.view(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x2 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
    x = x.view(x.shape[0], x.shape[1], 4,
               -1).permute(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
    x = self.norm(x) if self.norm else x
    x = self.reduction(x)
    out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
             (self.sampler.kernel_size[0] - 1) -
             1) // self.sampler.stride[0] + 1
    out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
             (self.sampler.kernel_size[1] - 1) -
             1) // self.sampler.stride[1] + 1

    output_size = (out_h, out_w)
    return x, output_size

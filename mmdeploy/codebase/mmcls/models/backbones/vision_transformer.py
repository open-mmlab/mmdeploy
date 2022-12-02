# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name=  # noqa: E251
    'mmcls.models.backbones.vision_transformer.VisionTransformer.forward',
    backend=Backend.NCNN.value)
def visiontransformer__forward__ncnn(self, x):
    """Rewrite `forward` of VisionTransformer for ncnn backend.

    The chunk in original VisionTransformer.forward will convert
    `self.cls_token` to `where` operator in ONNX, which will raise
    error in ncnn.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (VisionTransformer): The instance of the class InvertedResidual.
        x (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from InvertedResidual. The tensor
        shape (N, Cout, H, W).
    """
    from mmcls.models.utils import resize_pos_embed
    B = x.shape[0]
    x, patch_resolution = self.patch_embed(x)

    # cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((self.cls_token, x), dim=1)
    x = x + resize_pos_embed(
        self.pos_embed,
        self.patch_resolution,
        patch_resolution,
        mode=self.interpolate_mode,
        num_extra_tokens=self.num_extra_tokens)
    x = self.drop_after_pos(x)

    if not self.with_cls_token:
        # Remove class token for transformer encoder input
        x = x[:, 1:]

    outs = []
    for i, layer in enumerate(self.layers):
        x = layer(x)

        if i == len(self.layers) - 1 and self.final_norm:
            x = self.norm1(x)

        if i in self.out_indices:
            B, _, C = x.shape
            if self.with_cls_token:
                patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                cls_token = x[:, 0]
            else:
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                cls_token = None
            if self.output_cls_token:
                out = [patch_token, cls_token]
            else:
                out = patch_token
            outs.append(out)

    return tuple(outs)

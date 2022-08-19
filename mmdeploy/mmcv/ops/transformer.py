# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.transformer.PatchEmbed.forward',
    backend=Backend.NCNN.value)
def patch_embed__forward__ncnn(ctx, self, x):
    """Rewrite `forward` of PatchEmbed for ncnn backend.

    Args:
        x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

    Returns:
        tuple: Contains merged results and its spatial shape.

        - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
        - out_size (tuple[int]): Spatial shape of x, arrange as
            (out_h, out_w).
    """

    if self.adaptive_padding:
        x = self.adaptive_padding(x)

    x = self.projection(x)
    x_shape = x.shape
    out_size = (x_shape[2], x_shape[3])
    x = x.reshape((x_shape[0], x_shape[1], -1)).transpose(1, 2)
    if self.norm is not None:
        x = self.norm(x)
    return x, out_size

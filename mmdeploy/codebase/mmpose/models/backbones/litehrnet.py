# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.backbones.litehrnet.CrossResolutionWeighting.forward')
def cross_resolution_weighting__forward(ctx, self, x):
    """Rewrite ``forward`` for default backend.

    Rewrite this function to support export ``adaptive_avg_pool2d``.

    Args:
        x (list): block input.
    """

    mini_size = [int(_) for _ in x[-1].shape[-2:]]
    out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
    out = torch.cat(out, dim=1)
    out = self.conv1(out)
    out = self.conv2(out)
    out = torch.split(out, self.channels, dim=1)
    out = [
        s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
        for s, a in zip(x, out)
    ]
    return out

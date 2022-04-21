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


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.backbones.litehrnet.CrossResolutionWeighting.forward',
    backend='ncnn')
def cross_resolution_weighting__forward__ncnn(ctx, self, x):
    """Rewrite ``forward`` for ncnn backend.

    Rewrite this function to support export ``adaptive_avg_pool2d``
    for ncnn. Ncnn has its own implement of adaptive average pooling.

    Args:
        x (list): block input.
    """
    from mmdeploy.codebase.mmpose.core.ops import \
        ncnn_adaptive_avg_pool_forward

    mini_size = torch.tensor(x[-1].shape[-2:])
    # Replace torch.nn.functional.adaptive_avg_pool2d to a dummy onnx op,
    # which is used for ncnn implement.
    out = [ncnn_adaptive_avg_pool_forward(s, mini_size) for s in x[:-1]] + \
        [x[-1]]
    out = torch.cat(out, dim=1)
    out = self.conv1(out)
    out = self.conv2(out)
    out = torch.split(out, self.channels, dim=1)
    out = [
        s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
        for s, a in zip(x, out)
    ]
    return out


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.backbones.litehrnet.ShuffleUnit.forward', backend='ncnn')
def shuffle_unit__forward__ncnn(ctx, self, x):
    """Rewrite ``forward`` for ncnn backend.

    Rewrite this function to support export ``chunk``.

    Args:
        x (list): block input.
    """

    from mmpose.models.backbones.utils import channel_shuffle

    if self.stride > 1:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
    else:
        # Avoid `chunk` op for ncnn, which is not supported.
        channel = x.shape[1]
        c = channel // 2 + channel % 2
        x1 = x[:, 0:c, :, :]
        x2 = x[:, c:, :, :]
        out = torch.cat((x1, self.branch2(x2)), dim=1)

    out = channel_shuffle(out, 2)

    return out


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.backbones.litehrnet.ConditionalChannelWeighting.forward',
    backend='ncnn')
def conditional_channel_weighting_forward__ncnn(ctx, self, x):
    """Rewrite ``forward`` for ncnn backend.

    Rewrite this function to support export ``chunk``.

    Args:
        x (list): block input.
    """
    from mmpose.models.backbones.utils import channel_shuffle

    for i in range(len(x)):
        # Avoid `chunk` op for ncnn, which is not supported.
        channel = x[i].shape[1]
        c = channel // 2 + channel % 2
        x[i] = [x[i][:, 0:c, :, :], x[i][:, c:, :, :]]

    x1 = [s[0] for s in x]
    x2 = [s[1] for s in x]

    x2 = self.cross_resolution_weighting(x2)
    x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
    x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

    out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
    out = [channel_shuffle(s, 2) for s in out]

    return out


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.backbones.litehrnet.Stem.forward', backend='ncnn')
def stem__forward__ncnn(ctx, self, x):
    """Rewrite ``forward`` for ncnn backend.

    Rewrite this function to support export ``chunk``.

    Args:
        x (list): block input.
    """
    from mmpose.models.backbones.utils import channel_shuffle
    x = self.conv1(x)
    # Avoid `chunk` op for ncnn, which is not supported.
    channel = x.shape[1]
    c = channel // 2 + channel % 2
    x1 = x[:, 0:c, :, :]
    x2 = x[:, c:, :, :]

    x2 = self.expand_conv(x2)
    x2 = self.depthwise_conv(x2)
    x2 = self.linear_conv(x2)

    out = torch.cat((self.branch1(x1), x2), dim=1)
    out = channel_shuffle(out, 2)

    return out

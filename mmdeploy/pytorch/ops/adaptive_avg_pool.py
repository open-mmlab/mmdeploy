# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py

from torch.nn.modules.utils import _pair, _single, _triple
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER


def _adaptive_pool(name, type, tuple_fn, fn=None):
    """Generic adaptive pooling."""

    @parse_args('v', 'is')
    def symbolic_fn(g, input, output_size):
        if output_size == [1] * len(output_size) and type == 'AveragePool':
            return g.op('GlobalAveragePool', input)
        if not input.isCompleteTensor():
            if output_size == [1] * len(output_size):
                return g.op('GlobalMaxPool', input), None
            raise NotImplementedError(
                '[Adaptive pool]:input size not accessible')
        dim = input.type().sizes()[2:]
        if output_size == [1] * len(output_size) and type == 'MaxPool':
            return g.op('GlobalMaxPool', input), None

        # compute stride = floor(input_size / output_size)
        s = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]

        # compute kernel_size = input_size - (output_size - 1) * stride
        k = [dim[i] - (output_size[i] - 1) * s[i] for i in range(0, len(dim))]

        # call max_poolxd_with_indices to get indices in the output
        if type == 'MaxPool':
            return fn(g, input, k, k, (0, ) * len(dim), (1, ) * len(dim),
                      False)
        output = g.op(
            type,
            input,
            kernel_shape_i=tuple_fn(k),
            strides_i=tuple_fn(s),
            ceil_mode_i=False)
        return output

    return symbolic_fn


adaptive_avg_pool1d = _adaptive_pool('adaptive_avg_pool1d', 'AveragePool',
                                     _single)
adaptive_avg_pool2d = _adaptive_pool('adaptive_avg_pool2d', 'AveragePool',
                                     _pair)
adaptive_avg_pool3d = _adaptive_pool('adaptive_avg_pool3d', 'AveragePool',
                                     _triple)


@SYMBOLIC_REWRITER.register_symbolic('adaptive_avg_pool1d', is_pytorch=True)
def adaptive_avg_pool1d__default(ctx, *args):
    """Register default symbolic function for `adaptive_avg_pool1d`.

    Align symbolic of adaptive_pool between different torch version.
    """
    return adaptive_avg_pool1d(*args)


@SYMBOLIC_REWRITER.register_symbolic('adaptive_avg_pool2d', is_pytorch=True)
def adaptive_avg_pool2d__default(ctx, *args):
    """Register default symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_pool between different torch version.
    """
    return adaptive_avg_pool2d(*args)


@SYMBOLIC_REWRITER.register_symbolic('adaptive_avg_pool3d', is_pytorch=True)
def adaptive_avg_pool3d__default(ctx, *args):
    """Register default symbolic function for `adaptive_avg_pool3d`.

    Align symbolic of adaptive_pool between different torch version.
    """
    return adaptive_avg_pool3d(*args)


@SYMBOLIC_REWRITER.register_symbolic(
    'adaptive_avg_pool2d', is_pytorch=True, backend='ncnn')
def adaptive_avg_pool2d__ncnn(ctx, g, x, output_size):
    """Register ncnn symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_avg_pool2d in ncnn.
    """
    return g.op('mmdeploy::AdaptiveAvgPool2d', x, output_size)

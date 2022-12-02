# Copyright (c) OpenMMLab. All rights reserved.
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend, get_backend


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    """Symbolic function for `grid_sampler`.

    PyTorch does not support export grid_sampler to ONNX by default. We add the
    support here. `grid_sampler` will be exported as ONNX node
    'mmdeploy::grid_sampler'
    """
    return g.op(
        'mmdeploy::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler_ppl(g,
                     input,
                     grid,
                     interpolation_mode,
                     padding_mode,
                     align_corners=False):
    """Symbolic function for `grid_sampler`.

    PyTorch does not support export grid_sampler to ONNX by default. We add the
    support here. `grid_sampler` will be exported as ONNX node
    'mmdeploy::grid_sampler'
    """
    return g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@SYMBOLIC_REWRITER.register_symbolic('grid_sampler', is_pytorch=True)
def grid_sampler__default(*args):
    """Register default symbolic function for `grid_sampler`.

    Add support to grid_sample to ONNX.
    """
    ctx = SYMBOLIC_REWRITER.get_context()
    backend = get_backend(ctx.cfg)
    if backend == Backend.PPLNN:
        return grid_sampler_ppl(*args)
    else:
        return grid_sampler(*args)

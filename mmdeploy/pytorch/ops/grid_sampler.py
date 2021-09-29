from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REGISTER


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    """Symbolic function for `grid_sampler`."""
    return g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@SYMBOLIC_REGISTER.register_symbolic('grid_sampler', is_pytorch=True)
def grid_sampler_default(ctx, *args):
    """Register default symbolic function for `grid_sampler`."""
    return grid_sampler(*args)

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.interpolate', backend='ncnn')
def interpolate_static(ctx,
                       input,
                       size=None,
                       scale_factor=None,
                       mode='nearest',
                       align_corners=None,
                       recompute_scale_factor=None):
    """Rewrite `interpolate` for NCNN backend."""

    input_size = input.shape
    if scale_factor is None:
        scale_factor = [
            s_out / s_in for s_out, s_in in zip(size, input_size[2:])
        ]

    return ctx.origin_func(
        input,
        None,
        scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor)

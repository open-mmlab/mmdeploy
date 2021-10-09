from mmdeploy.core import SYMBOLIC_REGISTER


# Here using mmcv.ops.roi_align.__self__ to find
# mmcv.ops.roi_align.RoIAlignFunction, because RoIAlignFunction is not
# visible in mmcv.
@SYMBOLIC_REGISTER.register_symbolic(
    'mmcv.ops.roi_align.__self__', backend='default')
def roi_align_default(ctx, g, input, rois, output_size, spatial_scale,
                      sampling_ratio, pool_mode, aligned):
    """Rewrite symbolic function for default backend."""

    return g.op(
        'mmcv::MMCVRoiAlign',
        input,
        rois,
        output_height_i=output_size[0],
        output_width_i=output_size[1],
        spatial_scale_f=spatial_scale,
        sampling_ratio_i=sampling_ratio,
        mode_s=pool_mode,
        aligned_i=aligned)

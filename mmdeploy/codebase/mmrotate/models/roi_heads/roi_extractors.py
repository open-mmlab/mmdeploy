# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import RoIAlignRotated
from torch.autograd import Function

from mmdeploy.core.optimizers import mark
from mmdeploy.core.rewriters import FUNCTION_REWRITER


class MultiLevelRotatedRoiAlign(Function):
    """Create MMCVMultiLevelRotatedRoiAlign op.

    This class is used to create a MultiLevelRotatedRoiAlign in ONNX for the
    TensorRT backend.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def symbolic(g, *args):
        """Symbolic function for creating onnx op."""
        aligned = args[-1]
        featmap_strides = args[-2]
        finest_scale = args[-3]
        roi_scale_factor = args[-4]
        sampling_ratio = args[-5]
        clockwise = args[-6]
        output_size = args[-7]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]
        return g.op(
            'mmdeploy::MMCVMultiLevelRotatedRoiAlign',
            rois,
            *inputs,
            output_height_i=output_size[1],
            output_width_i=output_size[0],
            clockwise_i=clockwise,
            sampling_ratio_i=sampling_ratio,
            roi_scale_factor_f=roi_scale_factor,
            finest_scale_i=finest_scale,
            featmap_strides_f=featmap_strides,
            aligned_i=aligned)

    @staticmethod
    def forward(g, *args):
        """Run forward."""
        # aligned = args[-1]
        featmap_strides = args[-2]
        # finest_scale = args[-3]
        # roi_scale_factor = args[-4]
        # sampling_ratio = args[-5]
        output_size = args[-7]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]

        num_proposals = rois.shape[0]
        channel = inputs[0].shape[1]

        return rois.new_zeros(
            (num_proposals, channel, output_size[1], output_size[0]))


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.roi_extractors.'
    'rotate_single_level_roi_extractor.RotatedSingleRoIExtractor.forward',
    backend='tensorrt')
@mark(
    'rotated_roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
def rotated_single_roi_extractor__forward__tensorrt(ctx,
                                                    self,
                                                    feats,
                                                    rois,
                                                    roi_scale_factor=None):
    """Rewrite `forward` of `RotatedSingleRoIExtractor` for TensorRT backend.

    This function uses MMCVMultiLevelRoiAlign op for TensorRT deployment.
    """
    featmap_strides = self.featmap_strides
    finest_scale = self.finest_scale

    for roi_layer in self.roi_layers:
        assert isinstance(roi_layer, RoIAlignRotated
                          ), f'{type(roi_layer)} is not supported in TensorRT.'

    roi_layer = self.roi_layers[0]
    out_size = roi_layer.output_size
    sampling_ratio = roi_layer.sampling_ratio
    clockwise = roi_layer.clockwise
    aligned = roi_layer.aligned
    if roi_scale_factor is None:
        roi_scale_factor = 1.0

    featmap_strides = [float(s) for s in featmap_strides]
    return MultiLevelRotatedRoiAlign.apply(*feats, rois, out_size, clockwise,
                                           sampling_ratio, roi_scale_factor,
                                           finest_scale, featmap_strides,
                                           aligned)

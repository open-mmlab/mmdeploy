import torch
from torch.autograd import Function

from mmdeploy.core.optimizers import mark
from mmdeploy.core.rewriters import FUNCTION_REWRITER


class MultiLevelRoiAlign(Function):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def symbolic(g, *args):
        aligned = args[-1]
        featmap_strides = args[-2]
        finest_scale = args[-3]
        roi_scale_factor = args[-4]
        sampling_ratio = args[-5]
        output_size = args[-6]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]
        return g.op(
            'mmlab::MMCVMultiLevelRoiAlign',
            rois,
            *inputs,
            output_height_i=output_size[1],
            output_width_i=output_size[0],
            sampling_ratio_i=sampling_ratio,
            roi_scale_factor_f=roi_scale_factor,
            finest_scale_i=finest_scale,
            featmap_strides_f=featmap_strides,
            aligned_i=aligned)

    @staticmethod
    def forward(g, *args):
        # aligned = args[-1]
        featmap_strides = args[-2]
        # finest_scale = args[-3]
        # roi_scale_factor = args[-4]
        # sampling_ratio = args[-5]
        output_size = args[-6]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]

        num_proposals = rois.shape[0]
        channel = inputs[0].shape[1]

        return rois.new_zeros(
            (num_proposals, channel, output_size[1], output_size[0]))


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward',
    backend='tensorrt')
@mark('roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
def forward_of_single_roi_extractor_static(ctx,
                                           self,
                                           feats,
                                           rois,
                                           roi_scale_factor=None):
    """Rewrite `forward` for TensorRT backend."""
    featmap_strides = self.featmap_strides
    finest_scale = self.finest_scale

    roi_layer = self.roi_layers[0]
    out_size = roi_layer.output_size
    sampling_ratio = roi_layer.sampling_ratio
    aligned = roi_layer.aligned
    if roi_scale_factor is None:
        roi_scale_factor = 1.0

    featmap_strides = [float(s) for s in featmap_strides]
    return MultiLevelRoiAlign.apply(*feats, rois, out_size, sampling_ratio,
                                    roi_scale_factor, finest_scale,
                                    featmap_strides, aligned)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward')
@mark('roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
def forward_of_single_roi_extractor_dynamic(ctx,
                                            self,
                                            feats,
                                            rois,
                                            roi_scale_factor=None):
    """Rewrite `forward` for default backend."""
    out_size = self.roi_layers[0].output_size
    num_levels = len(feats)
    roi_feats = feats[0].new_zeros(rois.shape[0], self.out_channels, *out_size)
    if num_levels == 1:
        assert len(rois) > 0, 'The number of rois should be positive'
        return self.roi_layers[0](feats[0], rois)

    target_lvls = self.map_roi_levels(rois, num_levels)

    if roi_scale_factor is not None:
        rois = self.roi_rescale(rois, roi_scale_factor)

    for i in range(num_levels):
        mask = target_lvls == i
        inds = mask.nonzero(as_tuple=False).squeeze(1)

        # expand tensor to eliminate [0, ...] tensor
        rois_i = torch.cat((rois[inds], rois.new_zeros(1, 5)))

        roi_feats_t = self.roi_layers[i](feats[i], rois_i)

        # slice and recover the tensor
        roi_feats[inds] = roi_feats_t[0:-1]
    return roi_feats

from torch.autograd import Function

from mmdeploy.utils import FUNCTION_REWRITERS


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


@FUNCTION_REWRITERS.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward',
    backend='tensorrt')
def SingleRoIExtractor_forward_static(rewriter,
                                      self,
                                      feats,
                                      rois,
                                      roi_scale_factor=None):
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

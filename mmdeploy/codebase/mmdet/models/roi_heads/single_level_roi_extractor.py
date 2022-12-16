# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import RoIAlign
from torch.autograd import Function

from mmdeploy.core.optimizers import mark
from mmdeploy.core.rewriters import FUNCTION_REWRITER
from mmdeploy.utils import get_backend
from mmdeploy.utils.constants import Backend


class MultiLevelRoiAlign(Function):
    """Create MMCVMultiLevelRoiAlign op.

    This class is used to create a MultiLevelRoiAlign in ONNX for the TensorRT
    backend.
    """

    @staticmethod
    def symbolic(g, *args):
        """Symbolic function for creating onnx op."""
        aligned = args[-1]
        featmap_strides = args[-2]
        finest_scale = args[-3]
        roi_scale_factor = args[-4]
        sampling_ratio = args[-5]
        pool_mode = args[-6]
        pool_mode_flag = 0 if pool_mode == 'max' else 1
        output_size = args[-7]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]
        return g.op(
            'mmdeploy::MMCVMultiLevelRoiAlign',
            rois,
            *inputs,
            output_height_i=output_size[1],
            output_width_i=output_size[0],
            pool_mode_i=pool_mode_flag,
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


@mark('roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.roi_extractors.'
    'single_level_roi_extractor.SingleRoIExtractor.forward',
    backend='tensorrt')
def single_roi_extractor__forward__tensorrt(self,
                                            feats,
                                            rois,
                                            roi_scale_factor=None):
    """Rewrite `forward` of `SingleRoIExtractor` for TensorRT backend.

    This function uses MMCVMultiLevelRoiAlign op for TensorRT deployment.
    """
    featmap_strides = self.featmap_strides
    finest_scale = self.finest_scale

    for roi_layer in self.roi_layers:
        assert isinstance(
            roi_layer,
            RoIAlign), f'{type(roi_layer)} is not supported in TensorRT.'

    roi_layer = self.roi_layers[0]
    out_size = roi_layer.output_size
    sampling_ratio = roi_layer.sampling_ratio
    pool_mode = roi_layer.pool_mode
    aligned = roi_layer.aligned
    if roi_scale_factor is None:
        roi_scale_factor = 1.0

    featmap_strides = [float(s) for s in featmap_strides]
    return MultiLevelRoiAlign.apply(*feats, rois, out_size, pool_mode,
                                    sampling_ratio, roi_scale_factor,
                                    finest_scale, featmap_strides, aligned)


class AscendRoiExtractor(Function):
    """Create AscendRoiExtractor op.

    This class is used to create a AscendRoiExtractor in ONNX for the Ascend
    backend.
    """

    @staticmethod
    def symbolic(g, *args):
        """Symbolic function for creating onnx op."""
        aligned = args[-1]
        featmap_strides = [1 / stride for stride in args[-2]]
        finest_scale = args[-3]
        roi_scale_factor = args[-4]
        sampling_ratio = args[-5]
        pool_mode = args[-6]
        output_size = args[-7]
        inputs = args[:len(featmap_strides)]
        rois = args[len(featmap_strides)]

        return g.op(
            'mmdeploy::RoiExtractor',
            *inputs,
            rois,
            pooled_height_i=output_size[1],
            pooled_width_i=output_size[0],
            pool_mode_s=pool_mode,
            sample_num_i=sampling_ratio,
            roi_scale_factor_f=roi_scale_factor,
            finest_scale_i=finest_scale,
            spatial_scale_f=featmap_strides,
            aligned_i=aligned,
            outputs=1)

    @staticmethod
    def forward(ctx, *args):
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
    'mmdet.models.roi_heads.roi_extractors.'
    'single_level_roi_extractor.SingleRoIExtractor.forward',
    backend='ascend')
def single_roi_extractor__forward__ascend(self,
                                          feats,
                                          rois,
                                          roi_scale_factor=None):
    """Rewrite `forward` of `SingleRoIExtractor` for Ascend backend.

    This function uses RoiExtractor op for Ascend deployment.
    """
    featmap_strides = self.featmap_strides
    finest_scale = self.finest_scale

    for roi_layer in self.roi_layers:
        assert isinstance(
            roi_layer,
            RoIAlign), f'{type(roi_layer)} is not supported in Ascend.'

    roi_layer = self.roi_layers[0]
    out_size = roi_layer.output_size
    sampling_ratio = roi_layer.sampling_ratio
    pool_mode = roi_layer.pool_mode
    aligned = roi_layer.aligned
    if roi_scale_factor is None:
        roi_scale_factor = 1.0

    featmap_strides = [float(s) for s in featmap_strides]
    return AscendRoiExtractor.apply(*feats, rois, out_size, pool_mode,
                                    sampling_ratio, roi_scale_factor,
                                    finest_scale, featmap_strides, aligned)


@mark('roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward')
def single_roi_extractor__forward(self, feats, rois, roi_scale_factor=None):
    """Rewrite `forward` of SingleRoIExtractor for default backend.

    Rewrite this function to:
    1. enable exporting to IR even though the input
    image contains no targets. Note that, `ScatterND` of onnx may conflict with
    `Reshape` if a tensor have a dim size of 0. Thus, we have to cat zeros to
    the dim 0 of `roi_feats` and recover back after all roi align finished.

    2. this function adds mark for roi_extractor forward and remove
    unnecessary code of origin forward function when using ONNX as IR.

    3. use the roi align in torhcvision to accelerate the inference.
    """
    ctx = FUNCTION_REWRITER.get_context(
        'mmdet.models.roi_heads.SingleRoIExtractor.forward')
    backend = get_backend(ctx.cfg)
    out_size = self.roi_layers[0].output_size
    num_levels = len(feats)
    roi_feats = feats[0].new_zeros(rois.shape[0], self.out_channels, *out_size)
    if num_levels == 1:
        assert len(rois) > 0, 'The number of rois should be positive'
        if backend == Backend.TORCHSCRIPT or backend == Backend.COREML:
            self.roi_layers[0].use_torchvision = True
        return self.roi_layers[0](feats[0], rois)

    target_lvls = self.map_roi_levels(rois, num_levels)

    if roi_scale_factor is not None:
        rois = self.roi_rescale(rois, roi_scale_factor)

    # concate zeros to rois and roi_feats for empty tensor cases
    roi_feats = torch.cat(
        (roi_feats.new_zeros(num_levels * 2,
                             *roi_feats.shape[-3:]), roi_feats))
    rois = torch.cat((rois.new_zeros(num_levels * 2, 5), rois))
    _tmp = torch.linspace(
        0,
        num_levels - 1,
        num_levels,
        dtype=target_lvls.dtype,
        device=target_lvls.device)
    target_lvls = torch.cat((_tmp, _tmp, target_lvls))
    for i in range(num_levels):
        mask = target_lvls == i
        inds = mask.nonzero(as_tuple=False).squeeze(1)
        rois_t = rois[inds]
        # use the roi align in torhcvision
        if backend == Backend.TORCHSCRIPT or backend == Backend.COREML:
            self.roi_layers[i].use_torchvision = True
        roi_feats_t = self.roi_layers[i](feats[i], rois_t)
        roi_feats[inds] = roi_feats_t
    # slice to recover original size
    roi_feats = roi_feats[num_levels * 2:]
    return roi_feats


class SingleRoIExtractorOpenVINO(Function):
    """This class adds support for ExperimentalDetectronROIFeatureExtractor
    when exporting to OpenVINO.

    The `forward` method returns the original output, which is calculated in
    advance and added to the SingleRoIExtractorOpenVINO class. In addition, the
    list of arguments is changed here to be more suitable for
    ExperimentalDetectronROIFeatureExtractor.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(g, output_size, featmap_strides, sample_num, rois, *feats):
        """Run forward."""
        return SingleRoIExtractorOpenVINO.origin_output

    @staticmethod
    def symbolic(g, output_size, featmap_strides, sample_num, rois, *feats):
        """Symbolic function for creating onnx op."""
        from torch.onnx.symbolic_opset10 import _slice
        rois = _slice(g, rois, axes=[1], starts=[1], ends=[5])
        domain = 'org.openvinotoolkit'
        op_name = 'ExperimentalDetectronROIFeatureExtractor'
        roi_feats = g.op(
            f'{domain}::{op_name}',
            rois,
            *feats,
            output_size_i=output_size,
            pyramid_scales_i=featmap_strides,
            sampling_ratio_i=sample_num,
            image_id_i=0,
            distribute_rois_between_levels_i=1,
            preserve_rois_order_i=0,
            aligned_i=1,
            outputs=1)
        return roi_feats


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.roi_extractors.'
    'single_level_roi_extractor.SingleRoIExtractor.forward',
    backend='openvino')
def single_roi_extractor__forward__openvino(self,
                                            feats,
                                            rois,
                                            roi_scale_factor=None):
    """Replaces SingleRoIExtractor with SingleRoIExtractorOpenVINO when
    exporting to OpenVINO.

    This function uses ExperimentalDetectronROIFeatureExtractor for OpenVINO.
    """
    ctx = FUNCTION_REWRITER.get_context()

    # Adding original output to SingleRoIExtractorOpenVINO.
    state = torch._C._get_tracing_state()
    origin_output = ctx.origin_func(self, feats, rois, roi_scale_factor)
    setattr(SingleRoIExtractorOpenVINO, 'origin_output', origin_output)
    torch._C._set_tracing_state(state)

    output_size = self.roi_layers[0].output_size[0]
    featmap_strides = self.featmap_strides
    sample_num = self.roi_layers[0].sampling_ratio

    args = (output_size, featmap_strides, sample_num, rois, *feats)
    result = SingleRoIExtractorOpenVINO.apply(*args)
    return result


@mark('roi_extractor', inputs=['feats', 'rois'], outputs=['bbox_feats'])
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward',
    backend=Backend.TVM.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.SingleRoIExtractor.forward',
    backend=Backend.COREML.value)
def single_roi_extractor__forward__coreml(self,
                                          feats,
                                          rois,
                                          roi_scale_factor=None):
    """Rewrite `forward` of SingleRoIExtractor for coreml."""
    ctx = FUNCTION_REWRITER.get_context()
    backend = get_backend(ctx.cfg)
    out_size = self.roi_layers[0].output_size
    num_levels = len(feats)
    roi_feats = feats[0].new_zeros(rois.shape[0], self.out_channels, *out_size)
    if num_levels == 1:
        assert len(rois) > 0, 'The number of rois should be positive'
        self.roi_layers[0].use_torchvision = True
        return self.roi_layers[0](feats[0], rois)

    target_lvls = self.map_roi_levels(rois, num_levels)

    if roi_scale_factor is not None:
        rois = self.roi_rescale(rois, roi_scale_factor)

    for i in range(num_levels):
        mask = target_lvls == i
        # inds = mask.nonzero(as_tuple=False).squeeze(1)
        rois_t = rois * mask.unsqueeze(-1)
        # use the roi align in torhcvision
        if backend == Backend.COREML:
            self.roi_layers[i].use_torchvision = True
        roi_feats_t = self.roi_layers[i](feats[i], rois_t)
        roi_feats = roi_feats + roi_feats_t * (rois_t[:, -1] > 0).reshape(
            -1, 1, 1, 1)
    # slice to recover original size
    return roi_feats

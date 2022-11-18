# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures import DetDataSample
from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@mark(
    'instance_segmentor_forward',
    inputs=['input'],
    outputs=['dets', 'labels', 'masks'])
def __forward_impl_instance_seg(ctx, self, batch_inputs, data_samples,
                                **kwargs):
    """Rewrite and adding mark for `forward`.

    Encapsulate this function for rewriting `forward` of BaseDetector.
    1. Add mark for BaseDetector.
    2. Support both dynamic and static export to onnx.
    """
    x = self.extract_feat(batch_inputs)
    mask_outs = self.mask_head.predict(x, data_samples, rescale=False)
    return mask_outs


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.single_stage_instance_seg'
    '.SingleStageInstanceSegmentor.forward')
def single_stage_instance_segmentor__forward(
        ctx,
        self,
        batch_inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = 'tensor',
        **kwargs) -> ForwardResults:
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        batch_inputs (Tensor): Inputs with shape (N, C, H, W).
        data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        rescale (bool): Whether to rescale the results.
            Defaults to True.

    Returns:
        tuple[Tensor]: Detection results of the
        input images.
            - dets (Tensor): Classification bboxes and scores.
                Has a shape (num_instances, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
    """
    data_samples = copy.deepcopy(data_samples)
    if data_samples is None:
        data_samples = [DetDataSample()]
    deploy_cfg = ctx.cfg

    # get origin input shape as tensor to support onnx dynamic shape
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = torch._shape_as_tensor(batch_inputs)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]

    # set the metainfo
    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    return __forward_impl_instance_seg(
        ctx, self, batch_inputs, data_samples=data_samples, **kwargs)

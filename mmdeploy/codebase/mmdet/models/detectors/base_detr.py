# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures import DetDataSample
from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@mark('detr_predict', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def __predict_impl(self, batch_inputs, data_samples, rescale):
    """Rewrite and adding mark for `predict`.

    Encapsulate this function for rewriting `predict` of DetectionTransformer.
    1. Add mark for DetectionTransformer.
    2. Support both dynamic and static export to onnx.
    """
    img_feats = self.extract_feat(batch_inputs)
    head_inputs_dict = self.forward_transformer(img_feats, data_samples)
    results_list = self.bbox_head.predict(
        **head_inputs_dict, rescale=rescale, batch_data_samples=data_samples)
    return results_list


@torch.fx.wrap
def _set_metainfo(data_samples, img_shape):
    """Set the metainfo.

    Code in this function cannot be traced by fx.
    """

    # fx can not trace deepcopy correctly
    data_samples = copy.deepcopy(data_samples)
    if data_samples is None:
        data_samples = [DetDataSample()]

    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    return data_samples


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.base_detr.DetectionTransformer.forward')
def detection_transformer__forward(self,
                                   batch_inputs: torch.Tensor,
                                   data_samples: OptSampleList = None,
                                   rescale: bool = True,
                                   **kwargs) -> ForwardResults:
    """Rewrite `predict` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        batch_inputs (Tensor): Inputs with shape (N, C, H, W).
        data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        rescale (Boolean): rescale result or not.

    Returns:
        tuple[Tensor]: Detection results of the
        input images.
            - dets (Tensor): Classification bboxes and scores.
                Has a shape (num_instances, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
    """
    ctx = FUNCTION_REWRITER.get_context()

    deploy_cfg = ctx.cfg

    # get origin input shape as tensor to support onnx dynamic shape
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = torch._shape_as_tensor(batch_inputs)[2:].to(
        batch_inputs.device)
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]

    # set the metainfo
    data_samples = _set_metainfo(data_samples, img_shape)
    return __predict_impl(self, batch_inputs, data_samples, rescale)

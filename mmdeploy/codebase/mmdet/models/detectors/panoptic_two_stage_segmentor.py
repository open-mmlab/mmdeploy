# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.panoptic_two_stage_segmentor.'
    'TwoStagePanopticSegmentor.forward')
def two_stage_panoptic_segmentor__forward(self,
                                          batch_inputs,
                                          data_samples,
                                          mode='tensor',
                                          **kwargs):
    """Rewrite `forward` for default backend. Support configured dynamic/static
    shape for model input and return detection result as Tensor instead of
    numpy array.

    Args:
        batch_inputs (Tensor): Inputs with shape (N, C, H, W).
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        rescale (bool): Whether to rescale the results.
            Defaults to True.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            (bboxes, labels, masks, semseg), `bboxes` of shape [N, num_det, 5],
            `labels` of shape [N, num_det], `masks` of shape [N, roi_H, roi_W],
            `semseg` of shape [N, num_sem_class, sem_H, sem_W].
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
    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')
        data_sample.set_field(
            name='batch_input_shape', value=img_shape, field_type='metainfo')

    img_metas = [data_samples.metainfo for data_samples in data_samples]
    x = self.extract_feat(batch_inputs)
    if data_samples[0].get('proposals', None) is None:
        proposals = self.rpn_head.predict(x, data_samples, rescale=False)
    else:
        proposals = [data_sample.proposals for data_sample in data_samples]

    bboxes, labels, masks = self.roi_head.predict(
        x, proposals, data_samples, rescale=False)

    semseg = self.semantic_head.predict(x, img_metas, rescale=False)
    # do not export panoptic_fusion_head
    return bboxes, labels, masks, semseg

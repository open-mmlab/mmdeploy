# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape
from .single_stage import _set_metainfo


@mark("condinst_forward", inputs=["input"], outputs=["dets", "labels", "masks"])
def __forward_impl_condinst(self, batch_inputs, data_samples, **kwargs):
    """Rewrite and adding mark for `forward`.

    Encapsulate this function for rewriting `forward` of CondInst.
    1. Add mark for CondInst.
    2. Support both dynamic and static export to onnx.
    """
    # backbone
    x = self.extract_feat(batch_inputs)
    # bbox_head
    if self.with_bbox:
        results_list = self.bbox_head.predict(x, data_samples, rescale=False)
    else:
        results_list = None
    # mask_head
    mask_outs = self.mask_head.predict(
        x, data_samples, rescale=False, results_list=results_list
    )
    # prediction
    dets, labels, masks = mask_outs[0].dets.unsqueeze(0), mask_outs[0].labels.unsqueeze(0), mask_outs[0].masks.unsqueeze(0)

    return dets, labels, masks


@FUNCTION_REWRITER.register_rewriter(
    "mmdet.models.detectors.condinst.CondInst.forward", backend="default"
)
def condinst__forward(
    self,
    batch_inputs: torch.Tensor,
    data_samples: OptSampleList = None,
    mode: str = "tensor",
    **kwargs
) -> ForwardResults:
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
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg

    # get origin input shape as tensor to support onnx dynamic shape
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = torch._shape_as_tensor(batch_inputs)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]

    # set the metainfo
    data_samples = _set_metainfo(data_samples, img_shape)

    return __forward_impl_condinst(
        self, batch_inputs, data_samples=data_samples, **kwargs
    )

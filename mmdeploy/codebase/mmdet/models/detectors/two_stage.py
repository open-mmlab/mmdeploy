# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.two_stage.TwoStageDetector.extract_feat')
def two_stage_detector__extract_feat(self, img):
    """Rewrite `extract_feat` for default backend.

    This function uses the specific `extract_feat` function for the two
    stage detector after adding marks.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).

    Returns:
        list[Tensor]: Each item with shape (N, C, H, W) corresponds one
        level of backbone and neck features.
    """
    ctx = FUNCTION_REWRITER.get_context()

    @mark('extract_feat', inputs='img', outputs='feat')
    def __extract_feat_impl(self, img):
        return ctx.origin_func(self, img)

    return __extract_feat_impl(self, img)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.two_stage.TwoStageDetector.forward')
def two_stage_detector__forward(self,
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
        mode (str): export mode, not used.

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
    # note that we can not use `set_metainfo`, deepcopy would crash the
    # onnx trace.
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')

    x = self.extract_feat(batch_inputs)

    if data_samples[0].get('proposals', None) is None:
        rpn_results_list = self.rpn_head.predict(
            x, data_samples, rescale=False)
    else:
        rpn_results_list = [
            data_sample.proposals for data_sample in data_samples
        ]

    output = self.roi_head.predict(
        x, rpn_results_list, data_samples, rescale=False)
    return output

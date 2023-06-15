# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.structures import SegDataSample

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import get_codebase_config, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.BaseSegmentor.forward')
def base_segmentor__forward(self,
                            inputs,
                            data_samples=None,
                            mode='predict',
                            **kwargs):
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape for model input.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        inputs (Tensor | List[Tensor]): Input image tensor(s).
        data_samples (List[dict]): List of dicts containing image's meta
            information such as `img_shape`.

    Returns:
        torch.Tensor: Output segmentation map pf shape [N, 1, H, W].
    """

    # mark seg_input
    @mark('segmentor_forward', outputs=['input'])
    def __mark_input(inputs):
        return inputs

    inputs = __mark_input(inputs)

    ctx = FUNCTION_REWRITER.get_context()
    if data_samples is None:
        data_samples = [SegDataSample()]

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = inputs.shape[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    for data_sample in data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')
    seg_logit = self.predict(inputs, data_samples)

    # mark seg_head
    @mark('decode_head', outputs=['output'])
    def __mark_seg_logit(seg_logit):
        return seg_logit

    ctx = FUNCTION_REWRITER.get_context()
    with_argmax = get_codebase_config(ctx.cfg).get('with_argmax', True)
    # deal with out_channels=1 with two classes
    if seg_logit.shape[1] == 1:
        seg_logit = seg_logit.sigmoid()
        seg_pred = seg_logit > self.decode_head.threshold
        seg_pred = seg_pred.to(torch.int64)
    else:
        seg_pred = __mark_seg_logit(seg_logit)
        if with_argmax:
            seg_pred = seg_pred.argmax(dim=1, keepdim=True)
    return seg_pred

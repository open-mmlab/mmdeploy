# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.data import SegDataSample

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.BaseSegmentor.forward')
def base_segmentor__forward(ctx,
                            self,
                            batch_inputs,
                            batch_data_samples=None,
                            mode='predict',
                            **kwargs):
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape for model input.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_metas (List[dict]): List of dicts containing image's meta
            information such as `img_shape`.

    Returns:
        torch.Tensor: Output segmentation map pf shape [N, 1, H, W].
    """
    if batch_data_samples is None:
        batch_data_samples = [SegDataSample()]

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = batch_inputs.shape[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    for data_sample in batch_data_samples:
        data_sample.set_field(
            name='img_shape', value=img_shape, field_type='metainfo')
    return self.predict(batch_inputs, batch_data_samples)

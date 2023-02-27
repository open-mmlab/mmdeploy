# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.structures import SegDataSample

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


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
    return self.predict(inputs, data_samples)

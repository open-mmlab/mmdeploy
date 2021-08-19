import torch

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@mark(
    'detector_forward', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def _forward_of_base_detector_impl(ctx, self, img, img_metas=None, **kwargs):
    assert isinstance(img_metas, dict)
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(img)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.BaseDetector.forward')
def forward_of_base_detector(ctx, self, img, img_metas=None, **kwargs):
    if img_metas is None:
        img_metas = {}

    while isinstance(img_metas, list):
        img_metas = img_metas[0]

    if isinstance(img, list):
        img = torch.cat(img, 0)

    if 'return_loss' in kwargs:
        kwargs.pop('return_loss')
    return _forward_of_base_detector_impl(
        ctx, self, img, img_metas=img_metas, **kwargs)

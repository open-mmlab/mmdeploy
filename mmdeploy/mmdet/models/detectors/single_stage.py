import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.SingleStageDetector.extract_feat')
def extract_feat_of_single_stage(ctx, self, img):
    return ctx.origin_func(self, img)


@FUNCTION_REWRITER.register_rewriter(func_name='mmdet.models.RetinaNet.forward'
                                     )
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.SingleStageDetector.forward')
def forward_of_single_stage(ctx, self, data, **kwargs):
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(data)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    x = self.extract_feat(data)
    outs = self.bbox_head(x)
    return self.bbox_head.get_bboxes(*outs, img_shape, **kwargs)

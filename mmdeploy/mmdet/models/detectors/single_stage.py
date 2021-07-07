import torch

from mmdeploy.utils import FUNCTION_REWRITERS, is_dynamic_shape, mark


@FUNCTION_REWRITERS.register_rewriter(
    'mmdet.models.SingleStageDetector.extract_feat')
@mark('extract_feat')
def single_stage_extract_feat(rewriter, self, img):
    return rewriter.origin_func(self, img)


@FUNCTION_REWRITERS.register_rewriter(
    func_name='mmdet.models.RetinaNet.forward')
@FUNCTION_REWRITERS.register_rewriter(
    func_name='mmdet.models.SingleStageDetector.forward')
def single_stage_forward(rewriter, self, data, **kwargs):
    deploy_cfg = rewriter.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(data)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    x = self.extract_feat(data)
    outs = self.bbox_head(x)
    return self.bbox_head.get_bboxes(*outs, img_shape, **kwargs)

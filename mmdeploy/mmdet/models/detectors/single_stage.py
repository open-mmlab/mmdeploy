import torch

from mmdeploy.utils import FUNCTION_REWRITERS, mark


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
    # get origin input shape to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(data)[2:]
    x = self.extract_feat(data)
    outs = self.bbox_head(x)
    return self.bbox_head.get_bboxes(*outs, img_shape, **kwargs)

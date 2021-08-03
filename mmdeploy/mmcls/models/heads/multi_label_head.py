import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.MultiLabelClsHead.simple_test')
def simple_test_of_multi_label_head(ctx, self, cls_score, **kwargs):
    if isinstance(cls_score, list):
        cls_score = sum(cls_score) / float(len(cls_score))
    pred = F.sigmoid(cls_score) if cls_score is not None else None
    return pred

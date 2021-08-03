import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.StackedLinearClsHead.simple_test')
def simple_test_of_stacked_head(ctx, self, img, **kwargs):
    """Test without augmentation."""
    cls_score = img
    for layer in self.layers:
        cls_score = layer(cls_score)
    if isinstance(cls_score, list):
        cls_score = sum(cls_score) / float(len(cls_score))
    pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
    return pred

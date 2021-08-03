import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.MultiLabelLinearClsHead.simple_test')
def simple_test_of_multi_label_linear_head(ctx, self, img, **kwargs):
    """Test without augmentation."""
    cls_score = self.fc(img)
    if isinstance(cls_score, list):
        cls_score = sum(cls_score) / float(len(cls_score))
    pred = F.sigmoid(cls_score) if cls_score is not None else None
    return pred

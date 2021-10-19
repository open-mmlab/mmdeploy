import mmcv
import torch

from mmdeploy.core import RewriterContext
from mmdeploy.utils.test import WrapModel

input = torch.rand(1)


def test_baseclassfier_forward():
    from mmcls.models.classifiers import BaseClassifier

    class DummyClassifier(BaseClassifier):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg=init_cfg)

        def extract_feat(self, imgs):
            pass

        def forward_train(self, imgs):
            return 'train'

        def simple_test(self, img, tmp, **kwargs):
            return 'simple_test'

    model = DummyClassifier().eval()

    model_output = model(input)
    with RewriterContext(
            cfg=mmcv.Config(dict()), backend='onnxruntime'), torch.no_grad():
        backend_output = model(input)

    assert model_output == 'train'
    assert backend_output == 'simple_test'


def test_cls_head():
    from mmcls.models.heads.cls_head import ClsHead
    model = WrapModel(ClsHead(), 'post_process').eval()
    model_output = model(input)
    with RewriterContext(
            cfg=mmcv.Config(dict()), backend='onnxruntime'), torch.no_grad():
        backend_output = model(input)

    assert list(backend_output.detach().cpu().numpy()) == model_output


def test_multilabel_cls_head():
    from mmcls.models.heads.multi_label_head import MultiLabelClsHead
    model = WrapModel(MultiLabelClsHead(), 'post_process').eval()
    model_output = model(input)
    with RewriterContext(
            cfg=mmcv.Config(dict()), backend='onnxruntime'), torch.no_grad():
        backend_output = model(input)

    assert list(backend_output.detach().cpu().numpy()) == model_output

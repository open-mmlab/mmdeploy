# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.core import RewriterContext
from mmdeploy.utils.test import WrapModel, get_rewrite_outputs

input = torch.rand(1)


def get_invertedresudual_model():
    from mmcls.models.backbones.shufflenet_v2 import InvertedResidual
    model = InvertedResidual(16, 16)

    model.requires_grad_(False)
    return model


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


@pytest.mark.parametrize('backend_type',
                         ['onnxruntime', 'tensorrt', 'ncnn', 'openvino'])
def test_shufflenetv2_backbone__forward(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
    model = get_invertedresudual_model()
    model.cpu().eval()
    if backend_type == 'tensorrt':
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type,
                    common_config=dict(max_workspace_size=1 << 30),
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                input=dict(
                                    min_shape=[1, 16, 28, 28],
                                    opt_shape=[1, 16, 28, 28],
                                    max_shape=[1, 16, 28, 28])))
                    ]),
                onnx_config=dict(
                    input_shape=[28, 28], output_names=['output']),
                codebase_config=dict(type='mmcls', task='Classification')))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(type='mmcls', task='Classification')))

    imgs = torch.rand((1, 16, 28, 28))
    model_outputs = model.forward(imgs)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'x': imgs}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if isinstance(rewrite_outputs, dict):
        rewrite_outputs = rewrite_outputs['output']
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.cpu().numpy()
        if isinstance(rewrite_output, torch.Tensor):
            rewrite_output = rewrite_output.cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)

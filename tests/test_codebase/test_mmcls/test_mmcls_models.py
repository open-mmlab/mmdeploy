# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.core import RewriterContext
from mmdeploy.utils import Backend
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

input = torch.rand(1)


@pytest.fixture(scope='module')
def invertedresidual_model():
    from mmcls.models.backbones.shufflenet_v2 import InvertedResidual
    model = InvertedResidual(16, 16)

    model.requires_grad_(False)
    model.eval()
    return model


@pytest.fixture(scope='module')
def vit_model():
    from mmcls.models.classifiers.image import ImageClassifier
    model = ImageClassifier(
        backbone={
            'type':
            'VisionTransformer',
            'arch':
            'b',
            'img_size':
            384,
            'patch_size':
            32,
            'drop_rate':
            0.1,
            'init_cfg': [{
                'type': 'Kaiming',
                'layer': 'Conv2d',
                'mode': 'fan_in',
                'nonlinearity': 'linear'
            }]
        },
        head={
            'type': 'VisionTransformerClsHead',
            'num_classes': 1000,
            'in_channels': 768,
            'loss': {
                'type': 'CrossEntropyLoss',
                'loss_weight': 1.0
            },
            'topk': (1, 5)
        },
    )
    model.requires_grad_(False)
    model.eval()

    return model


def test_baseclassifier_forward():
    from mmcls.models.classifiers import BaseClassifier

    class DummyClassifier(BaseClassifier):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg=init_cfg)

        def extract_feat(self, imgs):
            pass

        def forward_train(self, imgs):
            return 'train'

        def simple_test(self, img, tmp=None, **kwargs):
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


@pytest.mark.parametrize(
    'backend_type',
    [Backend.ONNXRUNTIME, Backend.TENSORRT, Backend.NCNN, Backend.OPENVINO])
def test_shufflenetv2_backbone__forward(backend_type: Backend,
                                        invertedresidual_model):

    check_backend(backend_type, True)
    model = invertedresidual_model
    model.cpu().eval()
    if backend_type.value == 'tensorrt':
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type.value,
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
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(type='mmcls', task='Classification')))

    imgs = torch.rand((1, 16, 28, 28))
    model_outputs = model.forward(imgs)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'x': imgs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
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


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
def test_vision_transformer_backbone__forward(backend_type: Backend,
                                              vit_model):

    check_backend(backend_type, True)
    model = vit_model.eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmcls', task='Classification')))

    imgs = torch.rand((1, 3, 384, 384))
    model_outputs = model.forward(imgs, return_loss=False)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'img': imgs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if isinstance(rewrite_outputs, dict):
        rewrite_outputs = rewrite_outputs['output']
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        if isinstance(rewrite_output, torch.Tensor):
            rewrite_output = rewrite_output.cpu().numpy()
        assert np.allclose(
            model_output.reshape(-1),
            rewrite_output.reshape(-1),
            rtol=1e-03,
            atol=1e-05)


@pytest.mark.parametrize(
    'backend_type',
    [Backend.ONNXRUNTIME, Backend.TENSORRT, Backend.NCNN, Backend.OPENVINO])
@pytest.mark.parametrize('inputs',
                         [torch.rand(1, 3, 5, 5), (torch.rand(1, 3, 7, 7))])
def test_gap__forward(backend_type: Backend, inputs: list):
    check_backend(backend_type, False)

    from mmcls.models.necks import GlobalAveragePooling
    model = GlobalAveragePooling(dim=2)
    is_input_tensor = isinstance(inputs, torch.Tensor)
    if not is_input_tensor:
        assert len(inputs) == 1, 'only test one input'
    input_shape = inputs.shape if is_input_tensor else inputs[0].shape

    model.cpu().eval()
    if backend_type.value == 'tensorrt':
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type.value,
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                input=dict(
                                    min_shape=input_shape,
                                    opt_shape=input_shape,
                                    max_shape=input_shape)))
                    ]),
                onnx_config=dict(output_names=['output']),
                codebase_config=dict(type='mmcls', task='Classification')))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(type='mmcls', task='Classification')))

    inputs = torch.rand(input_shape)
    model_outputs = model(inputs)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'inputs': inputs if is_input_tensor else inputs[0]}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
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


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
@pytest.mark.parametrize('backend_type', [(Backend.TENSORRT)])
def test_shift_windows_msa_cls(backend_type: Backend):
    check_backend(backend_type)
    from mmcls.models.utils import ShiftWindowMSA
    model = ShiftWindowMSA(96, 3, 7)
    model.cuda().eval()
    output_names = ['output']

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(
                type=backend_type.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            query=dict(
                                min_shape=[1, 60800, 96],
                                opt_shape=[1, 60800, 96],
                                max_shape=[1, 60800, 96])))
                ]),
            onnx_config=dict(
                input_shape=None,
                input_names=['query'],
                output_names=output_names)))

    query = torch.randn([1, 60800, 96]).cuda()
    hw_shape = (torch.tensor(200), torch.tensor(304))

    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'query': query, 'hw_shape': hw_shape}
    _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

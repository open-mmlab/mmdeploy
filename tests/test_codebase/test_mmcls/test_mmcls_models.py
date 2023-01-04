# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.core.rewriters.rewriter_manager import RewriterContext
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close
try:
    import_codebase(Codebase.MMCLS)
except ImportError:
    pytest.skip(f'{Codebase.MMCLS} is not installed.', allow_module_level=True)

input = torch.rand(1)


def get_invertedresidual_model():
    from mmcls.models.backbones.shufflenet_v2 import InvertedResidual
    model = InvertedResidual(16, 16)

    model.requires_grad_(False)
    return model


def get_vit_model():
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

    return model


def test_baseclassifier_forward():
    from mmcls.models.classifiers import ImageClassifier

    from mmdeploy.codebase.mmcls import models  # noqa

    class DummyClassifier(ImageClassifier):

        def __init__(self, backbone):
            super().__init__(backbone=backbone)
            self.head = lambda x: x
            self.predict = lambda x, data_samples: x

        def extract_feat(self, batch_inputs: torch.Tensor):
            return batch_inputs

    input = torch.rand(1, 1000)
    backbone_cfg = dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch')
    model = DummyClassifier(backbone_cfg).eval()

    model_output = model(input, None, mode='predict')

    with RewriterContext({}):
        backend_output = model(input)

    torch_assert_close(model_output, input)
    torch_assert_close(backend_output, torch.nn.functional.softmax(input, -1))


@pytest.mark.parametrize(
    'backend_type',
    [Backend.ONNXRUNTIME, Backend.TENSORRT, Backend.NCNN, Backend.OPENVINO])
def test_shufflenetv2_backbone__forward(backend_type: Backend):

    check_backend(backend_type, True)
    model = get_invertedresidual_model()
    model.cpu().eval()
    if backend_type.value == 'tensorrt':
        deploy_cfg = Config(
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
        deploy_cfg = Config(
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
def test_vision_transformer_backbone__forward(backend_type: Backend):

    from mmcls.structures import ClsDataSample

    from mmdeploy.core import patch_model
    import_codebase(Codebase.MMCLS)
    check_backend(backend_type, True)
    model = get_vit_model()
    model.eval()

    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmcls', task='Classification')))

    imgs = torch.rand((1, 3, 384, 384))
    data_sample = ClsDataSample(
        metainfo=dict(
            scale_factor=(1, 1),
            ori_shape=imgs.shape[2:],
            img_shape=imgs.shape[2:]))
    model = patch_model(
        model, {}, backend=backend_type.value, data_samples=[data_sample])
    model_outputs = model.forward(imgs)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'batch_inputs': imgs}
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
            atol=1e-02)


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
        deploy_cfg = Config(
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
        deploy_cfg = Config(
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

    deploy_cfg = Config(
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

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv import ConfigDict

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, Task
from mmdeploy.utils.test import (WrapModel, check_backend, get_model_outputs,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMSEG)
except ImportError:
    pytest.skip(f'{Codebase.MMSEG} is not installed.', allow_module_level=True)

from mmseg.models import BACKBONES, HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@BACKBONES.register_module()
class ExampleBackbone(nn.Module):

    def __init__(self):
        super(ExampleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


@HEADS.register_module()
class ExampleDecodeHead(BaseDecodeHead):

    def __init__(self):
        super(ExampleDecodeHead, self).__init__(3, 3, num_classes=19)

    def forward(self, inputs):
        return self.cls_seg(inputs[0])


def get_model(type='EncoderDecoder',
              backbone='ExampleBackbone',
              decode_head='ExampleDecodeHead'):

    from mmseg.models import build_segmentor

    cfg = ConfigDict(
        type=type,
        backbone=dict(type=backbone),
        decode_head=dict(type=decode_head),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    segmentor = build_segmentor(cfg)

    return segmentor


def _demo_mm_inputs(input_shape=(1, 3, 8, 16), num_classes=10):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': 'horizontal'
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


@pytest.mark.parametrize('backend',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO, Backend.NCNN])
def test_encoderdecoder_simple_test(backend):
    check_backend(backend)
    segmentor = get_model()
    segmentor.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(output_names=['result'], input_shape=None),
            codebase_config=dict(type='mmseg', task='Segmentation')))

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    mm_inputs = _demo_mm_inputs(
        input_shape=(1, 3, 32, 32), num_classes=num_classes)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    model_inputs = {'img': imgs, 'img_meta': img_metas}
    model_outputs = get_model_outputs(segmentor, 'simple_test', model_inputs)
    img_meta = {
        'img_shape':
        (img_metas[0]['img_shape'][0], img_metas[0]['img_shape'][1])
    }
    wrapped_model = WrapModel(segmentor, 'simple_test', img_meta=img_meta)

    rewrite_inputs = {
        'img': imgs,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    model_outputs = torch.tensor(model_outputs[0])
    rewrite_outputs = rewrite_outputs[0].to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(rewrite_outputs, model_outputs)


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_basesegmentor_forward(backend):
    check_backend(backend)
    segmentor = get_model()
    segmentor.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(output_names=['result'], input_shape=None),
            codebase_config=dict(type='mmseg', task='Segmentation')))

    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    mm_inputs = _demo_mm_inputs(num_classes=num_classes)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    model_inputs = {
        'img': [imgs],
        'img_metas': [img_metas],
        'return_loss': False
    }
    model_outputs = get_model_outputs(segmentor, 'forward', model_inputs)

    wrapped_model = WrapModel(segmentor, 'forward')
    rewrite_inputs = {'img': imgs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    model_outputs = torch.tensor(model_outputs[0])
    rewrite_outputs = rewrite_outputs[0].to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(rewrite_outputs, model_outputs)


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_aspphead_forward(backend):
    check_backend(backend)
    from mmseg.models.decode_heads import ASPPHead
    head = ASPPHead(
        in_channels=32, channels=16, num_classes=19,
        dilations=(1, 12, 24)).eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(
                output_names=['result'], input_shape=(1, 32, 45, 45)),
            codebase_config=dict(type='mmseg', task='Segmentation')))
    inputs = [torch.randn(1, 32, 45, 45)]
    model_inputs = {'inputs': inputs}
    with torch.no_grad():
        model_outputs = get_model_outputs(head, 'forward', model_inputs)
    wrapped_model = WrapModel(head, 'forward')
    rewrite_inputs = {'inputs': inputs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]
    rewrite_outputs = rewrite_outputs.to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(
        rewrite_outputs, model_outputs, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO, Backend.NCNN])
def test_psphead_forward(backend):
    check_backend(backend)
    from mmseg.models.decode_heads import PSPHead
    head = PSPHead(in_channels=32, channels=16, num_classes=19).eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(output_names=['result'], input_shape=None),
            codebase_config=dict(type='mmseg', task='Segmentation')))
    inputs = [torch.randn(1, 32, 45, 45)]
    model_inputs = {'inputs': inputs}
    with torch.no_grad():
        model_outputs = get_model_outputs(head, 'forward', model_inputs)
    wrapped_model = WrapModel(head, 'forward')
    rewrite_inputs = {'inputs': inputs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]
    rewrite_outputs = rewrite_outputs.to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(rewrite_outputs, model_outputs, rtol=1, atol=1)


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_emamodule_forward(backend):
    check_backend(backend)
    from mmseg.models.decode_heads.ema_head import EMAModule
    head = EMAModule(8, 2, 2, 1.0).eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(
                output_names=['result'], input_shape=(1, 8, 16, 16)),
            codebase_config=dict(type='mmseg', task='Segmentation')))
    feats = torch.randn(1, 8, 16, 16)
    model_inputs = {'feats': feats}
    with torch.no_grad():
        model_outputs = get_model_outputs(head, 'forward', model_inputs)
    wrapped_model = WrapModel(head, 'forward')
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]
    rewrite_outputs = rewrite_outputs.to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(
        rewrite_outputs, model_outputs, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('is_dynamic_shape', [True, False])
@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_upconvblock_forward(backend, is_dynamic_shape):
    check_backend(backend)
    from mmseg.models.backbones.unet import BasicConvBlock
    from mmseg.models.utils import UpConvBlock

    head = UpConvBlock(BasicConvBlock, 16, 8, 8).eval()
    dynamic_axes = {
        'x': {
            0: 'b',
            2: 'h',
            3: 'w'
        },
        'skip': {
            0: 'b',
            2: 'h',
            3: 'w'
        },
        'output': {
            0: 'b',
            2: 'h',
            3: 'w'
        },
    } if is_dynamic_shape else None
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(
                input_names=['skip', 'x'],
                output_names=['output'],
                dynamic_axes=dynamic_axes),
            codebase_config=dict(
                type=Codebase.MMSEG.value, task=Task.SEGMENTATION.value)))
    x = torch.randn(1, 16, 16, 16)
    skip = torch.randn(1, 8, 32, 32)
    model_inputs = {'x': x, 'skip': skip}
    with torch.no_grad():
        model_outputs = get_model_outputs(head, 'forward', model_inputs)

    wrapped_model = WrapModel(head, 'forward')
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]
    rewrite_outputs = rewrite_outputs.to(model_outputs).reshape(
        model_outputs.shape)
    assert torch.allclose(
        rewrite_outputs, model_outputs, rtol=1e-03, atol=1e-05)

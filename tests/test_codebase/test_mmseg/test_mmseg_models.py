# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, Task
from mmdeploy.utils.test import (WrapModel, check_backend, get_model_outputs,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMSEG)
except ImportError:
    pytest.skip(f'{Codebase.MMSEG} is not installed.', allow_module_level=True)

from .utils import generate_datasample  # noqa: E402
from .utils import generate_mmseg_deploy_config  # noqa: E402
from .utils import generate_mmseg_task_processor  # noqa: E402


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_encoderdecoder_predict(backend):
    check_backend(backend)
    deploy_cfg = generate_mmseg_deploy_config(backend.value)
    task_processor = generate_mmseg_task_processor(deploy_cfg=deploy_cfg)
    segmentor = task_processor.build_pytorch_model()
    size = 256
    inputs = torch.randn(1, 3, size, size)
    data_samples = [generate_datasample(size, size)]
    wrapped_model = WrapModel(segmentor, 'predict', data_samples=data_samples)
    model_outputs = wrapped_model(inputs)[0].pred_sem_seg.data
    rewrite_inputs = {
        'inputs': inputs,
    }
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert torch.allclose(model_outputs, rewrite_outputs[0].squeeze(0))


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_basesegmentor_forward(backend):
    check_backend(backend)
    deploy_cfg = generate_mmseg_deploy_config(backend.value)
    task_processor = generate_mmseg_task_processor(deploy_cfg=deploy_cfg)
    segmentor = task_processor.build_pytorch_model()
    size = 256
    inputs = torch.randn(1, 3, size, size)
    data_samples = [generate_datasample(size, size)]
    wrapped_model = WrapModel(
        segmentor, 'forward', data_samples=data_samples, mode='predict')
    model_outputs = wrapped_model(inputs)[0].pred_sem_seg.data
    rewrite_inputs = {
        'inputs': inputs,
    }
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert torch.allclose(model_outputs, rewrite_outputs[0].squeeze(0))


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_emamodule_forward(backend):
    check_backend(backend)
    from mmseg.models.decode_heads.ema_head import EMAModule
    head = EMAModule(8, 2, 2, 1.0).eval()

    deploy_cfg = mmengine.Config(
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
    deploy_cfg = mmengine.Config(
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

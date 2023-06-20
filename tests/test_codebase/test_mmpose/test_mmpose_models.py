# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close

try:
    import_codebase(Codebase.MMPOSE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPOSE} is not installed.', allow_module_level=True)

from .utils import generate_mmpose_deploy_config  # noqa: E402
from .utils import generate_mmpose_task_processor  # noqa: E402


def get_heatmap_head():
    from mmpose.models.heads import HeatmapHead

    model = HeatmapHead(
        2,
        4,
        deconv_out_channels=(16, 16, 16),
        loss=dict(type='KeypointMSELoss', use_target_weight=False))
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_heatmaphead_forward(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_heatmap_head()
    model.cpu().eval()
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    feats = [torch.rand(1, 2, 32, 48)]
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'feats': feats}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    assert isinstance(rewrite_outputs, torch.Tensor)


def get_msmu_head():
    from mmpose.models.heads import MSPNHead
    model = MSPNHead(
        num_stages=1,
        num_units=1,
        out_shape=(32, 48),
        unit_channels=16,
        level_indices=[1])
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_msmuhead_forward(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_msmu_head()
    model.cpu().eval()
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    feats = [[torch.rand(1, 16, 32, 48)]]
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'feats': feats}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    assert isinstance(rewrite_outputs, torch.Tensor)


def get_cross_resolution_weighting_model():
    from mmpose.models.backbones.litehrnet import CrossResolutionWeighting

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.model = CrossResolutionWeighting([16, 16], ratio=8)

        def forward(self, x):
            assert isinstance(x, torch.Tensor)
            return self.model([x, x])

    model = DummyModel()
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_estimator_forward(backend_type: Backend):
    check_backend(backend_type, True)
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    task_processor = generate_mmpose_task_processor(deploy_cfg=deploy_cfg)
    model = task_processor.build_pytorch_model()
    model.requires_grad_(False)
    model.cpu().eval()
    wrapped_model = WrapModel(model, 'forward', data_samples=None)
    rewrite_inputs = {'inputs': torch.rand(1, 3, 256, 192)}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        run_with_backend=False,
        deploy_cfg=deploy_cfg)
    assert isinstance(rewrite_outputs, torch.Tensor)


def get_scale_norm_model():
    from mmpose.models.utils.rtmcc_block import ScaleNorm

    model = ScaleNorm(48)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
def test_scale_norm_forward(backend_type: Backend):
    check_backend(backend_type, True)
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    model = get_scale_norm_model()
    x = torch.rand(1, 17, 48)
    wrapped_model = WrapModel(model, 'forward')
    model_outputs = model.forward(x)
    rewrite_inputs = {'x': x}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    torch_assert_close(rewrite_outputs, model_outputs)


def get_rtmcc_block_model():
    from mmpose.models.utils.rtmcc_block import RTMCCBlock

    model = RTMCCBlock(48, 48, 48)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
def test_rtmcc_block_forward(backend_type: Backend):
    check_backend(backend_type, True)
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    model = get_rtmcc_block_model()
    inputs = torch.rand(1, 17, 48)
    wrapped_model = WrapModel(model, '_forward')
    model_outputs = model._forward(inputs)
    rewrite_inputs = {'inputs': inputs}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    torch_assert_close(rewrite_outputs, model_outputs)


def get_scale_model():
    from mmpose.models.utils.rtmcc_block import Scale

    model = Scale(48)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
def test_scale_forward(backend_type: Backend):
    check_backend(backend_type, True)
    deploy_cfg = generate_mmpose_deploy_config(backend_type.value)
    model = get_scale_model()
    x = torch.rand(1, 17, 48)
    wrapped_model = WrapModel(model, 'forward')
    model_outputs = model.forward(x)
    rewrite_inputs = {'x': x}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    torch_assert_close(rewrite_outputs, model_outputs)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_yolox_pose_head(backend_type: Backend):
    try:
        import models.yolox_pose_head
    except ImportError:
        pytest.skip('mmpose/projects/yolox-pose is not installed.', allow_module_level=True)

    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
    check_backend(backend_type, True)
    deploy_cfg, model_cfg = load_config(
        'configs/mmpose/yolox-pose_onnxruntime_static.py',
        'tests/test_codebase/test_mmpose/yolox-pose_s_8xb32-300e_coco.py'
    )
    task_processor = build_task_processor(model_cfg, deploy_cfg, device='cpu')
    model = task_processor.build_pytorch_model()
    model.cpu().eval()
    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(
        './demo/resources/human-pose.jpg',
        input_shape,
        data_preprocessor=getattr(model, 'data_preprocessor', None))
    pytorch_output = model(model_inputs)
    wrapped_model = WrapModel(model, 'forward')
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    rewrite_inputs = {'inputs': model_inputs}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        run_with_backend=False,
        deploy_cfg=deploy_cfg)
    torch_assert_close(rewrite_outputs, pytorch_output)

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

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

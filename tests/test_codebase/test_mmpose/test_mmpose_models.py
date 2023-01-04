# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend, Codebase, Task
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs


@pytest.fixture
def top_down_heatmap_simple_head_model():
    from mmpose.models.heads import TopdownHeatmapSimpleHead
    model = TopdownHeatmapSimpleHead(
        2,
        4,
        num_deconv_filters=(16, 16, 16),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False))
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.TENSORRT])
def test_top_down_heatmap_simple_head_inference_model(
        backend_type: Backend, top_down_heatmap_simple_head_model):
    check_backend(backend_type, True)
    model = top_down_heatmap_simple_head_model
    model.cpu().eval()
    if backend_type == Backend.TENSORRT:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type.value,
                    common_config=dict(max_workspace_size=1 << 30),
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                input=dict(
                                    min_shape=[1, 3, 32, 48],
                                    opt_shape=[1, 3, 32, 48],
                                    max_shape=[1, 3, 32, 48])))
                    ]),
                onnx_config=dict(
                    input_shape=[32, 48], output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    img = torch.rand((1, 2, 32, 48))
    model_outputs = model.inference_model(img)
    wrapped_model = WrapModel(model, 'inference_model')
    rewrite_inputs = {'x': img}
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
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.fixture
def top_down_heatmap_msmu_head_model():

    class DummyMSMUHead(torch.nn.Module):

        def __init__(self, out_shape):
            from mmpose.models.heads import TopdownHeatmapMSMUHead
            super().__init__()
            self.model = TopdownHeatmapMSMUHead(
                out_shape,
                unit_channels=2,
                out_channels=17,
                num_stages=1,
                num_units=1,
                loss_keypoint=dict(
                    type='JointsMSELoss', use_target_weight=False))

        def inference_model(self, x):
            assert isinstance(x, torch.Tensor)
            return self.model.inference_model([[x]], flip_pairs=None)

    model = DummyMSMUHead((32, 48))

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.TENSORRT])
def test_top_down_heatmap_msmu_head_inference_model(
        backend_type: Backend, top_down_heatmap_msmu_head_model):
    check_backend(backend_type, True)
    model = top_down_heatmap_msmu_head_model
    model.cpu().eval()
    if backend_type == Backend.TENSORRT:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type.value,
                    common_config=dict(max_workspace_size=1 << 30),
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                input=dict(
                                    min_shape=[1, 3, 32, 48],
                                    opt_shape=[1, 3, 32, 48],
                                    max_shape=[1, 3, 32, 48])))
                    ]),
                onnx_config=dict(
                    input_shape=[32, 48], output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    img = torch.rand((1, 2, 32, 48))
    model_outputs = model.inference_model(img)
    wrapped_model = WrapModel(model, 'inference_model')
    rewrite_inputs = {'x': img}
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
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.fixture
def cross_resolution_weighting_model():
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


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_cross_resolution_weighting_forward(backend_type: Backend,
                                            cross_resolution_weighting_model):
    check_backend(backend_type, True)
    model = cross_resolution_weighting_model
    model.cpu().eval()
    imgs = torch.rand(1, 16, 16, 16)

    if backend_type == Backend.NCNN:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value, use_vulkan=False),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    rewrite_inputs = {'x': imgs}
    model_outputs = model.forward(imgs)
    wrapped_model = WrapModel(model, 'forward')
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if isinstance(rewrite_outputs, dict):
        rewrite_outputs = rewrite_outputs['output']
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.cpu().numpy()
        if isinstance(rewrite_output, torch.Tensor):
            rewrite_output = rewrite_output.detach().cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.fixture
def top_down_model():
    from mmpose.models.detectors.top_down import TopDown
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(type='ResNet', depth=18),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=512,
            out_channels=17,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=False,
            post_process='default',
            shift_heatmap=False,
            modulate_kernel=11))
    model = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                    model_cfg['train_cfg'], model_cfg['test_cfg'],
                    model_cfg['pretrained'])

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.TENSORRT])
def test_top_down_forward(backend_type: Backend, top_down_model):
    check_backend(backend_type, True)
    model = top_down_model
    model.cpu().eval()
    if backend_type == Backend.TENSORRT:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type=backend_type.value,
                    common_config=dict(max_workspace_size=1 << 30),
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                input=dict(
                                    min_shape=[1, 3, 32, 32],
                                    opt_shape=[1, 3, 32, 32],
                                    max_shape=[1, 3, 32, 32])))
                    ]),
                onnx_config=dict(
                    input_shape=[32, 32], output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None, output_names=['output']),
                codebase_config=dict(
                    type=Codebase.MMPOSE.value,
                    task=Task.POSE_DETECTION.value)))
    img = torch.rand((1, 3, 32, 32))
    img_metas = {
        'image_file':
        'tests/test_codebase/test_mmpose' + '/data/imgs/dataset/blank.jpg',
        'center': torch.tensor([0.5, 0.5]),
        'scale': 1.,
        'location': torch.tensor([0.5, 0.5]),
        'bbox_score': 0.5
    }
    model_outputs = model.forward(
        img, img_metas=[img_metas], return_loss=False, return_heatmap=True)
    model_outputs = model_outputs['output_heatmap']
    wrapped_model = WrapModel(model, 'forward', return_loss=False)
    rewrite_inputs = {'img': img}
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
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)

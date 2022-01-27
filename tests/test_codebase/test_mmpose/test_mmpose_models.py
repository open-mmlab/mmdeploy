# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

import_codebase(Codebase.MMPOSE)

input = torch.rand(1)


class ListDummyMSMUHead(torch.nn.Module):

    def __init__(self,
                 out_shape,
                 unit_channels=256,
                 out_channels=17,
                 num_stages=4,
                 num_units=4,
                 use_prm=False,
                 norm_cfg=dict(type='BN'),
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        from mmpose.models.heads import TopdownHeatmapMSMUHead
        super().__init__()
        self.model = TopdownHeatmapMSMUHead(
            out_shape,
            unit_channels=unit_channels,
            out_channels=out_channels,
            num_stages=num_stages,
            num_units=num_units,
            use_prm=use_prm,
            norm_cfg=norm_cfg,
            loss_keypoint=loss_keypoint,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def inference_model(self, x, flip_pairs=None):
        assert len(x) == self.model.num_stages * self.model.num_units, \
            'the length of x should be' + \
            f'{self.model.num_stages * self.model.num_units}, got: {len(x)}'
        model_inputs = []
        for i in range(self.model.num_stages):
            stage_inputs = []
            for j in range(self.model.num_units):
                stage_inputs.append(x[i * self.model.num_units + j])
            model_inputs.append(stage_inputs)
        return self.model.inference_model(model_inputs, flip_pairs=flip_pairs)


def get_top_down_heatmap_msmu_head_model():
    model = ListDummyMSMUHead(
        (32, 48),
        unit_channels=2,
        num_stages=1,
        num_units=1,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False))

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_top_down_heatmap_msmu_head_inference_model(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_top_down_heatmap_msmu_head_model()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmpose', task='PoseDetection')))
    img = [[torch.rand((1, 2, 32, 48))]]
    flatten_img = []
    for stage in img:
        for unit in stage:
            flatten_img.append(unit)
    model_outputs = model.inference_model(flatten_img)
    wrapped_model = WrapModel(model, 'inference_model')
    rewrite_inputs = {'x': flatten_img}
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


def get_top_down_heatmap_simple_head_model():
    from mmpose.models.heads import TopdownHeatmapSimpleHead
    model = TopdownHeatmapSimpleHead(
        2,
        4,
        num_deconv_filters=(16, 16, 16),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False))
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_top_down_heatmap_simple_head_inference_model(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_top_down_heatmap_simple_head_model()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmpose', task='PoseDetection')))
    img = torch.rand((1, 2, 32, 48))
    flatten_img = []
    for stage in img:
        for unit in stage:
            flatten_img.append(unit)
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


def get_cross_resolution_weighting_model():
    from mmpose.models.backbones.litehrnet import CrossResolutionWeighting
    model = CrossResolutionWeighting([2, 4, 8, 16])

    model.requires_grad_(False)
    return model


def get_top_down_model():
    from mmpose.models.detectors.top_down import TopDown
    backbone_cfg = dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(4, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(2, 4)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(2, 4, 8)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(2, 4, 8, 16))),
    )
    channel_cfg = dict(
        num_output_channels=17,
        dataset_joints=17,
        dataset_channel=[[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])
    keypoint_cfg = dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
    test_cfg = dict(flip_test=False)
    model = TopDown(
        backbone_cfg, keypoint_head=keypoint_cfg, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_cross_resolution_weighting_forward(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_cross_resolution_weighting_model()
    model.cpu().eval()
    imgs = [
        torch.rand(1, 2, 16, 16),
        torch.rand(1, 4, 8, 8),
        torch.rand(1, 8, 4, 4),
        torch.rand(1, 16, 2, 2)
    ]
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmpose', task='PoseDetection')))
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
            rewrite_output = rewrite_output.cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_top_down_forward(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_top_down_model()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None, output_names=['output']),
            codebase_config=dict(type='mmpose', task='PoseDetection')))
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

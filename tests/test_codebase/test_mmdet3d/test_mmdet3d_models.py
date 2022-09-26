# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, Task, load_config
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

try:
    import_codebase(Codebase.MMDET3D)
except ImportError:
    pytest.skip(
        f'{Codebase.MMDET3D} is not installed.', allow_module_level=True)
model_cfg = load_config(
    'tests/test_codebase/test_mmdet3d/data/model_cfg.py')[0]
monodet_model_cfg = load_config(
    'tests/test_codebase/test_mmdet3d/data/monodet_model_cfg.py')[0]


def get_pillar_encoder():
    from mmdet3d.models.voxel_encoders import PillarFeatureNet
    model = PillarFeatureNet(
        in_channels=4,
        feat_channels=(64, ),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max')
    model.requires_grad_(False)
    return model


def get_pointpillars_scatter():
    from mmdet3d.models.middle_encoders import PointPillarsScatter
    model = PointPillarsScatter(in_channels=64, output_shape=(16, 16))
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_pillar_encoder(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_pillar_encoder()
    model.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                input_names=['features', 'num_points', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))
    features = torch.rand(3945, 32, 4) * 100
    num_points = torch.randint(0, 32, (3945, ), dtype=torch.int32)
    coors = torch.randint(0, 10, (3945, 4), dtype=torch.int32)
    model_outputs = [model.forward(features, num_points, coors)]
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {
        'features': features,
        'num_points': num_points,
        'coors': coors
    }
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
            model_output.shape, rewrite_output.shape, rtol=1e-03, atol=1e-03)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_pointpillars_scatter(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_pointpillars_scatter()
    model.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                input_names=['voxel_features', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))
    voxel_features = torch.rand(16 * 16, 64) * 100
    coors = torch.randint(0, 10, (16 * 16, 4), dtype=torch.int32)
    model_outputs = [model.forward_batch(voxel_features, coors, 1)]
    wrapped_model = WrapModel(model, 'forward_batch')
    rewrite_inputs = {'voxel_features': voxel_features, 'coors': coors}
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
            model_output.shape, rewrite_output.shape, rtol=1e-03, atol=1e-03)


def get_centerpoint():
    from mmdet3d.models.detectors.centerpoint import CenterPoint

    model = CenterPoint(**model_cfg.centerpoint_model)
    model.requires_grad_(False)
    return model


def get_centerpoint_head():
    from mmdet3d.models import builder
    model_cfg.centerpoint_model.pts_bbox_head.test_cfg = model_cfg.\
        centerpoint_model.test_cfg
    head = builder.build_head(model_cfg.centerpoint_model.pts_bbox_head)
    head.requires_grad_(False)
    return head


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_centerpoint(backend_type: Backend):
    from mmdeploy.codebase.mmdet3d.deploy.voxel_detection import VoxelDetection
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)
    model = get_centerpoint()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['voxels', 'num_points', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))
    voxeldetection = VoxelDetection(model_cfg, deploy_cfg, 'cpu')
    inputs, data = voxeldetection.create_input(
        'tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin')

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        outputs = model.forward(*data)
        head = get_centerpoint_head()
        rewrite_outputs = head.get_bboxes(*[[i] for i in outputs],
                                          inputs['img_metas'][0])
    assert rewrite_outputs is not None


def get_pointpillars_nus():
    from mmdet3d.models.detectors import MVXFasterRCNN

    model = MVXFasterRCNN(**model_cfg.pointpillars_nus_model)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_pointpillars_nus(backend_type: Backend):
    from mmdeploy.codebase.mmdet3d.deploy.voxel_detection import VoxelDetection
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)
    model = get_pointpillars_nus()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['voxels', 'num_points', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))
    voxeldetection = VoxelDetection(model_cfg, deploy_cfg, 'cpu')
    inputs, data = voxeldetection.create_input(
        'tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin')

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        outputs = model.forward(*data)
    assert outputs is not None


def get_fcos3d():
    from mmdet3d.models.detectors import FCOSMono3D
    monodet_model_cfg.model.pop('type')
    model = FCOSMono3D(**monodet_model_cfg.model)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_fcos3d(backend_type: Backend):
    from mmdeploy.codebase.mmdet3d.deploy.monocular_detection import \
        MonocularDetection
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)
    model = get_fcos3d()
    model.cpu().eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['img', 'cam2img', 'cam2img_inverse'],
                output_names=[
                    'bboxes', 'scores', 'labels', 'dir_scores', 'attrs'
                ],
            ),
            codebase_config=dict(
                type=Codebase.MMDET3D.value,
                task=Task.MONOCULAR_DETECTION.value,
                ann_file='tests/test_codebase/test_mmdet3d/data/nuscenes'
                '/n015-2018-07-24-11-22-45+0800__CAM_BACK__'
                '1532402927637525_mono3d.coco.json')))
    monoculardetection = MonocularDetection(monodet_model_cfg, deploy_cfg,
                                            'cpu')
    data, inputs = monoculardetection.create_input(
        'tests/test_codebase/test_mmdet3d/data/nuscenes/n015-2018-07-24-'
        '11-22-45+0800__CAM_BACK__1532402927637525.jpg')

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        outputs = model.forward(*inputs, img_metas=data['img_metas'])
    assert outputs is not None

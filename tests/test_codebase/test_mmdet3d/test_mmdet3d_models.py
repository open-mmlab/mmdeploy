# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import numpy as np
import pytest
import torch

from mmdeploy.apis import build_task_processor
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

    deploy_cfg = mmengine.Config(
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
    model_outputs = model.forward(features, num_points, coors)
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
    if isinstance(rewrite_outputs, list):
        rewrite_outputs = rewrite_outputs[0]

    assert np.allclose(
        model_outputs.shape, rewrite_outputs.shape, rtol=1e-03, atol=1e-03)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_pointpillars_scatter(backend_type: Backend):
    check_backend(backend_type, True)
    model = get_pointpillars_scatter()
    model.cpu().eval()

    deploy_cfg = mmengine.Config(
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
    model_outputs = model.forward_batch(voxel_features, coors, 1)
    wrapped_model = WrapModel(model, 'forward_batch')
    rewrite_inputs = {'voxel_features': voxel_features, 'coors': coors}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if isinstance(rewrite_outputs, list):
        rewrite_outputs = rewrite_outputs[0]
    assert np.allclose(
        model_outputs.shape, rewrite_outputs.shape, rtol=1e-03, atol=1e-03)


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
def test_pointpillars(backend_type: Backend):
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)

    model_cfg = load_config(
        'tests/test_codebase/test_mmdet3d/data/model_cfg.py')[0]
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['voxels', 'num_points', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))

    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    model = task_processor.build_pytorch_model(None)
    model.eval()

    preproc = task_processor.build_data_preprocessor()
    _, data = task_processor.create_input(
        pcd='tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin',
        data_preprocessor=preproc)

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        outputs = model.forward(*data)
        assert len(outputs) == 3


def get_pointpillars_nus():
    from mmdet3d.models.detectors import MVXFasterRCNN

    model = MVXFasterRCNN(**model_cfg.pointpillars_nus_model)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_centerpoint(backend_type: Backend):
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)

    centerpoint_model_cfg = load_config(
        'tests/test_codebase/test_mmdet3d/data/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'  # noqa: E501
    )[0]

    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['voxels', 'num_points', 'coors'],
                output_names=['outputs']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))

    task_processor = build_task_processor(centerpoint_model_cfg, deploy_cfg,
                                          'cpu')
    model = task_processor.build_pytorch_model(None)
    model.eval()

    preproc = task_processor.build_data_preprocessor()
    _, data = task_processor.create_input(
        pcd=  # noqa: E251
        'tests/test_codebase/test_mmdet3d/data/nuscenes/n008-2018-09-18-12-07-26-0400__LIDAR_TOP__1537287083900561.pcd.bin',  # noqa: E501
        data_preprocessor=preproc)

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        outputs = model.forward(data)
    assert outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_smoke(backend_type: Backend):
    from mmdeploy.core import RewriterContext
    check_backend(backend_type, True)

    model_cfg = load_config(
        'tests/test_codebase/test_mmdet3d/data/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'  # noqa: E501
    )[0]  # noqa: E501
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                opset_version=11,
                input_names=['input'],
                output_names=['cls_score', 'bbox_pred']),
            codebase_config=dict(
                type=Codebase.MMDET3D.value, task=Task.MONO_DETECTION.value)))

    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    model = task_processor.build_pytorch_model(None)
    model.eval()

    preproc = task_processor.build_data_preprocessor()
    _, data = task_processor.create_input(
        pcd=  # noqa: E251
        'tests/test_codebase/test_mmdet3d/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl',  # noqa: E501
        data_preprocessor=preproc)

    with RewriterContext(
            cfg=deploy_cfg,
            backend=deploy_cfg.backend_config.type,
            opset=deploy_cfg.onnx_config.opset_version):
        cls_score, bbox_pred = model.forward(data)
        assert len(cls_score) == 1 and len(bbox_pred) == 1

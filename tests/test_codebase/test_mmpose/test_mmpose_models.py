# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import pytest
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

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
        from mmyolo.utils.setup_env import register_all_modules
        from models.yolox_pose_head import YOLOXPoseHead  # noqa: F401,F403
        register_all_modules(True)
    except ImportError:
        pytest.skip(
            'mmpose/projects/yolox-pose is not installed.',
            allow_module_level=True)
    deploy_cfg = mmengine.Config.fromfile(
        'configs/mmpose/pose-detection_yolox-pose_onnxruntime_dynamic.py')
    check_backend(backend_type, True)

    head = YOLOXPoseHead(
        head_module=dict(
            type='YOLOXPoseHeadModule',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            widen_factor=0.5,
            stacked_convs=2,
            num_keypoints=17,
            featmap_strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_pose=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=30.0),
        loss_bbox_aux=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0),
        train_cfg=ConfigDict(
            assigner=dict(
                type='PoseSimOTAAssigner',
                center_radius=2.5,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                oks_calculator=dict(
                    type='OksLoss',
                    metainfo='configs/_base_/datasets/coco.py'))),
        test_cfg=ConfigDict(
            yolox_style=True,
            multi_label=False,
            score_thr=0.001,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.65)))

    class TestYOLOXPoseHeadModel(torch.nn.Module):

        def __init__(self, yolox_pose_head):
            super(TestYOLOXPoseHeadModel, self).__init__()
            self.yolox_pose_head = yolox_pose_head

        def forward(self, x1, x2, x3):
            inputs = [x1, x2, x3]
            data_sample = InstanceData()
            data_sample.set_metainfo(
                dict(ori_shape=(640, 640), scale_factor=(1.0, 1.0)))
            return self.yolox_pose_head.predict(
                inputs, batch_data_samples=[data_sample])

    model = TestYOLOXPoseHeadModel(head)
    model.cpu().eval()

    model_inputs = [
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 128, 4, 4),
        torch.randn(1, 128, 2, 2)
    ]

    with torch.no_grad():
        pytorch_output = model(*model_inputs)[0]
    pred_bboxes = torch.from_numpy(pytorch_output.bboxes).unsqueeze(0)
    pred_bboxes_scores = torch.from_numpy(pytorch_output.scores).reshape(
        1, -1, 1)
    pred_kpts = torch.from_numpy(pytorch_output.keypoints).unsqueeze(0)
    pred_kpts_scores = torch.from_numpy(
        pytorch_output.keypoint_scores).unsqueeze(0).unsqueeze(-1)

    pytorch_output = [
        torch.cat([pred_bboxes, pred_bboxes_scores], dim=-1),
        torch.cat([pred_kpts, pred_kpts_scores], dim=-1)
    ]

    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {
        'x1': model_inputs[0],
        'x2': model_inputs[1],
        'x3': model_inputs[2]
    }
    deploy_cfg.onnx_config.input_names = ['x1', 'x2', 'x3']

    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        run_with_backend=True,
        deploy_cfg=deploy_cfg)

    # keep bbox coord >= 0
    rewrite_outputs[0] = rewrite_outputs[0].clamp(min=0)
    torch_assert_close(rewrite_outputs, pytorch_output)

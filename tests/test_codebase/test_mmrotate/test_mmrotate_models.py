# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
from typing import Dict, List

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.config_utils import get_ir_config
from mmdeploy.utils.test import (WrapModel, check_backend, get_model_outputs,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMROTATE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMROTATE} is not installed.', allow_module_level=True)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def convert_to_list(rewrite_output: Dict, output_names: List[str]) -> List:
    """Converts output from a dictionary to a list.

    The new list will contain only those output values, whose names are in list
    'output_names'.
    """
    outputs = [
        value for name, value in rewrite_output.items() if name in output_names
    ]
    return outputs


def get_anchor_head_model():
    """AnchorHead Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))

    from mmrotate.models.dense_heads import RotatedAnchorHead
    model = RotatedAnchorHead(num_classes=4, in_channels=1, test_cfg=test_cfg)
    model.requires_grad_(False)

    return model


def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize(
    'model_cfg_path',
    ['tests/test_codebase/test_mmrotate/data/single_stage_model.json'])
def test_forward_of_base_detector(model_cfg_path, backend):
    check_backend(backend)
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(
                output_names=['dets', 'labels'], input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    pre_top_k=-1,
                    keep_top_k=100,
                ))))

    model_cfg = mmcv.Config(dict(model=mmcv.load(model_cfg_path)))
    model_cfg.model = _replace_r50_with_r18(model_cfg.model)

    from mmrotate.models import build_detector

    model_cfg.model.pretrained = None
    model_cfg.model.train_cfg = None
    model = build_detector(model_cfg.model, test_cfg=model_cfg.get('test_cfg'))
    model.cfg = model_cfg
    model.to('cpu')

    img = torch.randn(1, 3, 64, 64)
    rewrite_inputs = {'img': img}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None


def get_deploy_cfg(backend_type: Backend, ir_type: str):
    return mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                type=ir_type,
                output_names=['dets', 'labels'],
                input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000,
                ))))


@pytest.mark.parametrize('backend_type, ir_type',
                         [(Backend.ONNXRUNTIME, 'onnx')])
def test_base_dense_head_get_bboxes(backend_type: Backend, ir_type: str):
    """Test get_bboxes rewrite of base dense head."""
    check_backend(backend_type)
    anchor_head = get_anchor_head_model()
    anchor_head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    deploy_cfg = get_deploy_cfg(backend_type, ir_type)
    output_names = get_ir_config(deploy_cfg).get('output_names', None)

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 45, 32, 32), (1, 45, 16, 16),
    # (1, 45, 8, 8), (1, 45, 4, 4), (1, 45, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 36, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 45, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(anchor_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        anchor_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        if isinstance(rewrite_outputs, dict):
            rewrite_outputs = convert_to_list(rewrite_outputs, output_names)
        for model_output, rewrite_output in zip(model_outputs[0],
                                                rewrite_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            # hard code to make two tensors with the same shape
            # rewrite and original codes applied different nms strategy
            assert np.allclose(
                model_output[:rewrite_output.shape[0]][:2],
                rewrite_output[:2],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_single_roi_extractor():
    """SingleRoIExtractor Config."""
    from mmrotate.models.roi_heads import RotatedSingleRoIExtractor
    roi_layer = dict(
        type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True)
    out_channels = 1
    featmap_strides = [4, 8, 16, 32]
    model = RotatedSingleRoIExtractor(roi_layer, out_channels,
                                      featmap_strides).eval()

    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_rotated_single_roi_extractor(backend_type: Backend):
    check_backend(backend_type)

    single_roi_extractor = get_single_roi_extractor()
    output_names = ['roi_feat']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
            )))

    seed_everything(1234)
    out_channels = single_roi_extractor.out_channels
    feats = [
        torch.rand((1, out_channels, 200, 336)),
        torch.rand((1, out_channels, 100, 168)),
        torch.rand((1, out_channels, 50, 84)),
        torch.rand((1, out_channels, 25, 42)),
    ]
    seed_everything(5678)
    rois = torch.tensor(
        [[0.0000, 587.8285, 52.1405, 886.2484, 341.5644, 0.0000]])

    model_inputs = {
        'feats': feats,
        'rois': rois,
    }
    model_outputs = get_model_outputs(single_roi_extractor, 'forward',
                                      model_inputs)

    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=single_roi_extractor,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    if isinstance(backend_outputs, dict):
        backend_outputs = backend_outputs.values()
    for model_output, backend_output in zip(model_outputs[0], backend_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        backend_output = backend_output.squeeze()
        assert np.allclose(
            model_output, backend_output, rtol=1e-03, atol=1e-05)


def get_oriented_rpn_head_model():
    """Oriented RPN Head Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))
    from mmrotate.models.dense_heads import OrientedRPNHead
    model = OrientedRPNHead(
        in_channels=1,
        version='le90',
        bbox_coder=dict(type='MidpointOffsetCoder', angle_range='le90'),
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_get_bboxes_of_oriented_rpn_head(backend_type: Backend):
    check_backend(backend_type)
    head = get_oriented_rpn_head_model()
    head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['dets', 'labels']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 54, 32, 32), (1, 54, 16, 16),
    # (1, 54, 8, 8), (1, 54, 4, 4), (1, 54, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 9, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 54, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


def get_rotated_rpn_head_model():
    """Oriented RPN Head Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))
    from mmrotate.models.dense_heads import RotatedRPNHead
    model = RotatedRPNHead(
        version='le90',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_get_bboxes_of_rotated_rpn_head(backend_type: Backend):
    check_backend(backend_type)
    head = get_rotated_rpn_head_model()
    head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['dets', 'labels']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000))))

    # the cls_score's size: (1, 3, 32, 32), (1, 3, 16, 16),
    # (1, 3, 8, 8), (1, 3, 4, 4), (1, 3, 2, 2).
    # the bboxes's size: (1, 18, 32, 32), (1, 18, 16, 16),
    # (1, 18, 8, 8), (1, 18, 4, 4), (1, 18, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 3, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 18, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_rotate_standard_roi_head__simple_test(backend_type: Backend):
    check_backend(backend_type)
    from mmrotate.models.roi_heads import OrientedStandardRoIHead
    output_names = ['dets', 'labels']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000))))
    angle_version = 'le90'
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))
    head = OrientedStandardRoIHead(
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=3,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=3,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True),
        test_cfg=test_cfg)
    head.cpu().eval()

    seed_everything(1234)
    x = [torch.rand(1, 3, pow(2, i), pow(2, i)) for i in range(4, 0, -1)]
    proposals = [torch.rand(1, 100, 6), torch.randint(0, 10, (1, 100))]
    img_metas = [{'img_shape': torch.tensor([224, 224])}]

    wrapped_model = WrapModel(
        head, 'simple_test', proposals=proposals, img_metas=img_metas)
    rewrite_inputs = {'x': x}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_gv_ratio_roi_head__simple_test(backend_type: Backend):
    check_backend(backend_type)
    from mmrotate.models.roi_heads import GVRatioRoIHead
    output_names = ['dets', 'labels']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000,
                    max_output_boxes_per_class=1000))))
    angle_version = 'le90'
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))
    head = GVRatioRoIHead(
        version=angle_version,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=3,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='GVBBoxHead',
            version=angle_version,
            num_shared_fcs=2,
            in_channels=3,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            ratio_thr=0.8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2)),
            fix_coder=dict(type='GVFixCoder', angle_range=angle_version),
            ratio_coder=dict(type='GVRatioCoder', angle_range=angle_version),
            reg_class_agnostic=True),
        test_cfg=test_cfg)
    head.cpu().eval()

    seed_everything(1234)
    x = [torch.rand(1, 3, pow(2, i), pow(2, i)) for i in range(4, 0, -1)]
    bboxes = torch.rand(1, 100, 2)
    bboxes = torch.cat(
        [bboxes, bboxes + torch.rand(1, 100, 2) + torch.rand(1, 100, 1)],
        dim=-1)
    proposals = [bboxes, torch.randint(0, 10, (1, 100))]
    img_metas = [{'img_shape': torch.tensor([224, 224])}]

    wrapped_model = WrapModel(
        head, 'simple_test', proposals=proposals, img_metas=img_metas)
    rewrite_inputs = {'x': x}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


def get_roi_trans_roi_head_model():
    """Oriented RPN Head Config."""
    angle_version = 'le90'

    num_stages = 2
    stage_loss_weights = [1, 1]
    version = angle_version
    bbox_roi_extractor = [
        dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
    ]

    bbox_head = [
        dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAHBBoxCoder',
                angle_range=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ]
    test_cfg = mmcv.Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))

    args = [num_stages, stage_loss_weights, bbox_roi_extractor, bbox_head]
    kwargs = {'version': version, 'test_cfg': test_cfg}

    from mmrotate.models.roi_heads import RoITransRoIHead
    model = RoITransRoIHead(*args, **kwargs).eval()
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_simple_test_of_roi_trans_roi_head(backend_type: Backend):
    check_backend(backend_type)

    roi_head = get_roi_trans_roi_head_model()
    roi_head.cpu()

    seed_everything(1234)
    x = [
        torch.rand((1, 64, 32, 32)),
        torch.rand((1, 64, 16, 16)),
        torch.rand((1, 64, 8, 8)),
        torch.rand((1, 64, 4, 4)),
    ]
    proposals = torch.tensor([[[58.8285, 52.1405, 188.2484, 141.5644, 0.5]]])
    labels = torch.tensor([[[0.]]])
    s = 256
    img_metas = [{
        'img_shape': torch.tensor([s, s]),
        'ori_shape': torch.tensor([s, s]),
        'scale_factor': torch.tensor([1, 1, 1, 1])
    }]

    model_inputs = {
        'x': x,
    }

    output_names = ['det_bboxes', 'det_labels']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=2000,
                    keep_top_k=2000))))

    wrapped_model = WrapModel(
        roi_head,
        'simple_test',
        proposal_list=[proposals, labels],
        img_metas=img_metas)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None

# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Dict, List

import numpy as np
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
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
    test_cfg = Config(
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


def get_deploy_cfg(backend_type: Backend, ir_type: str):
    return Config(
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
    check_backend(backend_type, True)

    single_roi_extractor = get_single_roi_extractor()
    output_names = ['roi_feat']
    deploy_cfg = Config(
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
    test_cfg = Config(
        dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.1),
            max_per_img=2000))
    from mmrotate.models.dense_heads import OrientedRPNHead
    model = OrientedRPNHead(
        in_channels=1,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(type='MidpointOffsetCoder', angle_version='le90'),
        test_cfg=test_cfg,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0))

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_oriented_rpn_head__predict_by_feat(backend_type: Backend):
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
    deploy_cfg = Config(
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
        torch.rand(1, 3, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 18, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head, 'predict_by_feat', batch_img_metas=img_metas, with_nms=True)
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
def test_gv_ratio_roi_head__predict_bbox(backend_type: Backend):
    check_backend(backend_type, True)
    from mmrotate.models.roi_heads import GVRatioRoIHead
    output_names = ['dets', 'labels']
    deploy_cfg = Config(
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
    test_cfg = Config(
        dict(
            rcnn=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms_rotated', iou_threshold=0.1),
                max_per_img=2000)))
    head = GVRatioRoIHead(
        bbox_roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=3,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='GVBBoxHead',
            num_shared_fcs=2,
            in_channels=3,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            ratio_thr=0.8,
            bbox_coder=dict(
                type='DeltaXYWHQBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2)),
            fix_coder=dict(type='GVFixCoder'),
            ratio_coder=dict(type='GVRatioCoder'),
            predict_box_type='rbox',
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_fix=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 3.0, loss_weight=1.0),
            loss_ratio=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 3.0, loss_weight=16.0),
        ))
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
        head,
        'predict_bbox',
        rpn_results_list=proposals,
        batch_img_metas=img_metas,
        rcnn_test_cfg=test_cfg['rcnn'])
    rewrite_inputs = {'x': x}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_gvfixcoder__decode(backend_type: Backend):
    check_backend(backend_type)

    deploy_cfg = Config(
        dict(
            onnx_config=dict(output_names=['output'], input_shape=None),
            backend_config=dict(type=backend_type.value),
            codebase_config=dict(type='mmrotate', task='RotatedDetection')))

    from mmrotate.models.task_modules.coders import GVFixCoder
    coder = GVFixCoder()

    hboxes = torch.rand(1, 10, 4)
    fix_deltas = torch.rand(1, 10, 4)

    wrapped_model = WrapModel(coder, 'decode')
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model,
        model_inputs={
            'hboxes': hboxes,
            'fix_deltas': fix_deltas
        },
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None


def get_rotated_rtmdet_head_model():
    """RTMDet-R Head Config."""
    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000))

    from mmrotate.models.dense_heads import RotatedRTMDetHead
    model = RotatedRTMDetHead(
        num_classes=4,
        in_channels=1,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_rotated_rtmdet_head_predict_by_feat(backend_type: Backend):
    """Test predict_by_feat rewrite of RTMDet-R."""
    check_backend(backend_type, require_plugin=True)
    rtm_r_head = get_rotated_rtmdet_head_model()
    rtm_r_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]
    output_names = ['dets', 'labels']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=3000,
                    keep_top_k=2000,
                    max_output_boxes_per_class=2000))))
    seed_everything(1234)
    cls_scores = [
        torch.rand(1, rtm_r_head.num_classes, 2 * pow(2, i), 2 * pow(2, i))
        for i in range(3, 0, -1)
    ]
    seed_everything(5678)
    bbox_preds = [
        torch.rand(1, 4, 2 * pow(2, i), 2 * pow(2, i))
        for i in range(3, 0, -1)
    ]
    seed_everything(9101)
    angle_preds = [
        torch.rand(1, rtm_r_head.angle_coder.encode_size, 2 * pow(2, i),
                   2 * pow(2, i)) for i in range(3, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'angle_preds': angle_preds,
        'batch_img_metas': batch_img_metas,
        'with_nms': True
    }
    model_outputs = get_model_outputs(rtm_r_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        rtm_r_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'angle_preds': angle_preds,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        # hard code to make two tensors with the same shape
        # rewrite and original codes applied different nms strategy
        min_shape = min(model_outputs[0].bboxes.shape[0],
                        rewrite_outputs[0].shape[1], 5)
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes.tensor[:min_shape],
                rewrite_outputs[0][i, :min_shape, :5],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 5],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None

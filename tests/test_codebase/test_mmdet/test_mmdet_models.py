# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
from typing import Dict, List

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapModel, backend_checker, check_backend,
                                 get_model_outputs, get_rewrite_outputs)


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
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models import AnchorHead
    model = AnchorHead(num_classes=4, in_channels=1, test_cfg=test_cfg)
    model.requires_grad_(False)

    return model


def get_fcos_head_model():
    """FCOS Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models import FCOSHead
    model = FCOSHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_rpn_head_model():
    """RPN Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            nms_pre=0,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0))
    from mmdet.models import RPNHead
    model = RPNHead(in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_single_roi_extractor():
    """SingleRoIExtractor Config."""
    from mmdet.models.roi_heads import SingleRoIExtractor
    roi_layer = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
    out_channels = 1
    featmap_strides = [4, 8, 16, 32]
    model = SingleRoIExtractor(roi_layer, out_channels, featmap_strides).eval()

    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.NCNN, Backend.OPENVINO])
def test_anchor_head_get_bboxes(backend_type: Backend):
    """Test get_bboxes rewrite of anchor head."""
    check_backend(backend_type)
    anchor_head = get_anchor_head_model()
    anchor_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 36, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 36, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

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
        anchor_head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
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
                model_output[:rewrite_output.shape[0]],
                rewrite_output,
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.NCNN, Backend.OPENVINO])
def test_get_bboxes_of_fcos_head(backend_type: Backend):
    check_backend(backend_type)
    fcos_head = get_fcos_head_model()
    fcos_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, fcos_head.num_classes, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    seed_everything(9101)
    centernesses = [
        torch.rand(1, 1, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'centernesses': centernesses,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(fcos_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        fcos_head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'centernesses': centernesses
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        if isinstance(rewrite_outputs, dict):
            rewrite_outputs = [
                value for name, value in rewrite_outputs.items()
                if name in output_names
            ]
        for model_output, rewrite_output in zip(model_outputs[0],
                                                rewrite_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            # hard code to make two tensors with the same shape
            # rewrite and original codes applied different nms strategy
            assert np.allclose(
                model_output[:rewrite_output.shape[0]],
                rewrite_output,
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_get_bboxes_of_rpn_head(backend_type: Backend):
    check_backend(backend_type)
    head = get_rpn_head_model()
    head.cpu().eval()
    s = 4
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['dets']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 9, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 36, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    assert rewrite_outputs is not None


def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model


@pytest.mark.parametrize('model_cfg_path', [
    'tests/test_codebase/test_mmdet/data/single_stage_model.json',
    'tests/test_codebase/test_mmdet/data/mask_model.json'
])
@backend_checker(Backend.ONNXRUNTIME)
def test_forward_of_base_detector(model_cfg_path):
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(
                output_names=['dets', 'labels'], input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    model_cfg = mmcv.Config(dict(model=mmcv.load(model_cfg_path)))
    model_cfg.model = _replace_r50_with_r18(model_cfg.model)
    from mmdet.apis import init_detector
    model = init_detector(model_cfg, None, 'cpu')

    img = torch.randn(1, 3, 64, 64)
    rewrite_inputs = {'img': img}
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_single_roi_extractor(backend_type: Backend):
    check_backend(backend_type)

    single_roi_extractor = get_single_roi_extractor()
    output_names = ['roi_feat']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
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
    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

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


def get_cascade_roi_head(is_instance_seg=False):
    """CascadeRoIHead Config."""
    num_stages = 3
    stage_loss_weights = [1, 0.5, 0.25]
    bbox_roi_extractor = {
        'type': 'SingleRoIExtractor',
        'roi_layer': {
            'type': 'RoIAlign',
            'output_size': 7,
            'sampling_ratio': 0
        },
        'out_channels': 64,
        'featmap_strides': [4, 8, 16, 32]
    }
    all_target_stds = [[0.1, 0.1, 0.2, 0.2], [0.05, 0.05, 0.1, 0.1],
                       [0.033, 0.033, 0.067, 0.067]]
    bbox_head = [{
        'type': 'Shared2FCBBoxHead',
        'in_channels': 64,
        'fc_out_channels': 1024,
        'roi_feat_size': 7,
        'num_classes': 80,
        'bbox_coder': {
            'type': 'DeltaXYWHBBoxCoder',
            'target_means': [0.0, 0.0, 0.0, 0.0],
            'target_stds': target_stds
        },
        'reg_class_agnostic': True,
        'loss_cls': {
            'type': 'CrossEntropyLoss',
            'use_sigmoid': False,
            'loss_weight': 1.0
        },
        'loss_bbox': {
            'type': 'SmoothL1Loss',
            'beta': 1.0,
            'loss_weight': 1.0
        }
    } for target_stds in all_target_stds]

    mask_roi_extractor = {
        'type': 'SingleRoIExtractor',
        'roi_layer': {
            'type': 'RoIAlign',
            'output_size': 14,
            'sampling_ratio': 0
        },
        'out_channels': 64,
        'featmap_strides': [4, 8, 16, 32]
    }
    mask_head = {
        'type': 'FCNMaskHead',
        'num_convs': 4,
        'in_channels': 64,
        'conv_out_channels': 64,
        'num_classes': 80,
        'loss_mask': {
            'type': 'CrossEntropyLoss',
            'use_mask': True,
            'loss_weight': 1.0
        }
    }

    test_cfg = mmcv.Config(
        dict(
            score_thr=0.05,
            nms=mmcv.Config(dict(type='nms', iou_threshold=0.5)),
            max_per_img=100,
            mask_thr_binary=0.5))

    args = [num_stages, stage_loss_weights, bbox_roi_extractor, bbox_head]
    kwargs = {'test_cfg': test_cfg}
    if is_instance_seg:
        args += [mask_roi_extractor, mask_head]

    from mmdet.models import CascadeRoIHead
    model = CascadeRoIHead(*args, **kwargs).eval()
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_cascade_roi_head(backend_type: Backend):
    check_backend(backend_type)

    cascade_roi_head = get_cascade_roi_head()
    seed_everything(1234)
    x = [
        torch.rand((1, 64, 200, 304)),
        torch.rand((1, 64, 100, 152)),
        torch.rand((1, 64, 50, 76)),
        torch.rand((1, 64, 25, 38)),
    ]
    proposals = torch.tensor([[587.8285, 52.1405, 886.2484, 341.5644, 0.5]])
    img_metas = mmcv.Config({
        'img_shape': torch.tensor([800, 1216]),
        'ori_shape': torch.tensor([800, 1216]),
        'scale_factor': torch.tensor([1, 1, 1, 1])
    })

    model_inputs = {
        'x': x,
        'proposal_list': [proposals],
        'img_metas': [img_metas]
    }
    model_outputs = get_model_outputs(cascade_roi_head, 'simple_test',
                                      model_inputs)
    processed_model_outputs = []
    outputs = model_outputs[0]
    for output in outputs:
        if output.shape == (0, 5):
            processed_model_outputs.append(np.zeros((1, 5)))
        else:
            processed_model_outputs.append(output)
    processed_model_outputs = np.array(processed_model_outputs).squeeze()
    processed_model_outputs = processed_model_outputs[None, :, :]

    output_names = ['results']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1))))
    model_inputs = {'x': x, 'proposals': proposals.unsqueeze(0)}
    wrapped_model = WrapModel(
        cascade_roi_head, 'simple_test', img_metas=img_metas)
    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)

    if isinstance(backend_outputs, (list, tuple)) and \
            backend_outputs[0].shape == (1, 0, 5):
        processed_backend_outputs = torch.zeros((1, 80, 5))
    else:
        processed_backend_outputs = backend_outputs

    model_output = processed_model_outputs
    backend_output = [
        out.detach().cpu().numpy() for out in processed_backend_outputs
    ]
    assert np.allclose(model_output, backend_output, rtol=1e-03, atol=1e-05)


def get_fovea_head_model():
    """FoveaHead Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models import FoveaHead
    model = FoveaHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_get_bboxes_of_fovea_head(backend_type: Backend):
    check_backend(backend_type)
    fovea_head = get_fovea_head_model()
    fovea_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, fovea_head.num_classes, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(fovea_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(fovea_head, 'get_bboxes', img_metas=img_metas[0])
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
                model_output[:rewrite_output.shape[0]],
                rewrite_output,
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_atss_head_model():
    """ATSSHead Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    anchor_generator = dict(
        type='AnchorGenerator',
        ratios=[1.0],
        octave_base_scale=8,
        scales_per_octave=1,
        strides=[8, 16, 32, 64, 128])
    from mmdet.models import ATSSHead
    model = ATSSHead(
        num_classes=4,
        in_channels=1,
        test_cfg=test_cfg,
        anchor_generator=anchor_generator)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_get_bboxes_of_atss_head(backend_type):
    check_backend(backend_type)
    atss_head = get_atss_head_model()
    atss_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, atss_head.num_classes, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    seed_everything(9101)
    centernesses = [
        torch.rand(1, 1, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'centernesses': centernesses,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(atss_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        atss_head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'centernesses': centernesses
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
                model_output[:rewrite_output.shape[0]],
                rewrite_output,
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.OPENVINO])
def test_cascade_roi_head_with_mask(backend_type: Backend):
    check_backend(backend_type)

    cascade_roi_head = get_cascade_roi_head(is_instance_seg=True)
    seed_everything(1234)
    x = [
        torch.rand((1, 64, 200, 304)),
        torch.rand((1, 64, 100, 152)),
        torch.rand((1, 64, 50, 76)),
        torch.rand((1, 64, 25, 38)),
    ]
    proposals = torch.tensor([[587.8285, 52.1405, 886.2484, 341.5644, 0.5]])
    img_metas = mmcv.Config({
        'img_shape': torch.tensor([800, 1216]),
        'ori_shape': torch.tensor([800, 1216]),
        'scale_factor': torch.tensor([1, 1, 1, 1])
    })

    output_names = ['bbox_results', 'segm_results']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1))))
    model_inputs = {'x': x, 'proposals': proposals.unsqueeze(0)}
    wrapped_model = WrapModel(
        cascade_roi_head, 'simple_test', img_metas=img_metas)
    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    bbox_results = backend_outputs[0]
    segm_results = backend_outputs[1]
    expected_bbox_results = np.zeros((1, 80, 5))
    expected_segm_results = -np.ones((1, 80))
    assert np.allclose(
        expected_bbox_results, bbox_results, rtol=1e-03,
        atol=1e-05), 'bbox_results do not match.'
    assert np.allclose(
        expected_segm_results, segm_results, rtol=1e-03,
        atol=1e-05), 'segm_results do not match.'


def get_yolov3_head_model():
    """yolov3 Head Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100))
    from mmdet.models import YOLOV3Head
    model = YOLOV3Head(
        num_classes=4,
        in_channels=[16, 8, 4],
        out_channels=[32, 16, 8],
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.NCNN, Backend.OPENVINO])
def test_yolov3_head_get_bboxes(backend_type):
    """Test get_bboxes rewrite of yolov3 head."""
    check_backend(backend_type)
    yolov3_head = get_yolov3_head_model()
    yolov3_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.45,
                    confidence_threshold=0.005,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    seed_everything(1234)
    pred_maps = [
        torch.rand(1, 27, 5, 5),
        torch.rand(1, 27, 10, 10),
        torch.rand(1, 27, 20, 20)
    ]
    # to get outputs of pytorch model
    model_inputs = {'pred_maps': pred_maps, 'img_metas': img_metas}
    model_outputs = get_model_outputs(yolov3_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolov3_head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'pred_maps': pred_maps,
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
                model_output[:rewrite_output.shape[0]],
                rewrite_output,
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_yolox_head_model():
    """YOLOX Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models import YOLOXHead
    model = YOLOXHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_yolox_head_get_bboxes(backend_type: Backend):
    """Test get_bboxes rewrite of YOLOXHead."""
    check_backend(backend_type)
    yolox_head = get_yolox_head_model()
    yolox_head.cpu().eval()
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=20,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                ))))
    seed_everything(1234)
    cls_scores = [
        torch.rand(1, yolox_head.num_classes, pow(2, i), pow(2, i))
        for i in range(3, 0, -1)
    ]
    seed_everything(5678)
    bbox_preds = [
        torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(3, 0, -1)
    ]
    seed_everything(9101)
    objectnesses = [
        torch.rand(1, 1, pow(2, i), pow(2, i)) for i in range(3, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(yolox_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolox_head, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
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
            min_shape = min(model_output.shape[0], rewrite_output.shape[0], 20)
            assert np.allclose(
                model_output[:min_shape],
                rewrite_output[:min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_vfnet_head_model():
    """VFNet Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    from mmdet.models import VFNetHead
    model = VFNetHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    model.cpu().eval()
    return model


@pytest.mark.parametrize('backend_type', [Backend.OPENVINO])
def test_get_bboxes_of_vfnet_head(backend_type: Backend):
    """Test get_bboxes rewrite of VFNet head."""
    check_backend(backend_type)

    class TestModel(torch.nn.Module):
        """Stub for VFNetHead with fake bbox_preds operations.

        Then bbox_preds will be one of the inputs to the ONNX graph.
        """

        def __init__(self, vfnet_head):
            super().__init__()
            self.vfnet_head = vfnet_head

        def get_bboxes(self,
                       cls_scores,
                       bbox_preds,
                       bbox_preds_refine,
                       img_metas,
                       cfg=None,
                       rescale=None,
                       with_nms=True):
            tmp_bbox_pred_refine = []
            for bbox_pred, bbox_pred_refine in zip(bbox_preds,
                                                   bbox_preds_refine):
                tmp = bbox_pred_refine + bbox_pred
                tmp = tmp - bbox_pred
                tmp_bbox_pred_refine.append(tmp)
            bbox_preds_refine = tmp_bbox_pred_refine
            return self.vfnet_head.get_bboxes(cls_scores, bbox_preds,
                                              bbox_preds_refine, img_metas,
                                              cfg, rescale, with_nms)

    test_model = TestModel(get_vfnet_head_model())
    test_model.requires_grad_(False)
    test_model.cpu().eval()

    s = 16
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
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    seed_everything(1234)
    cls_score = [
        torch.rand(1, test_model.vfnet_head.num_classes, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]
    seed_everything(9101)
    bbox_preds_refine = [
        torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]

    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'bbox_preds_refine': bbox_preds_refine,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(test_model, 'get_bboxes', model_inputs)

    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        test_model, 'get_bboxes', img_metas=img_metas[0], with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'bbox_preds_refine': bbox_preds_refine
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
            min_shape = min(model_output.shape[0], rewrite_output.shape[0])
            assert np.allclose(
                model_output[:min_shape],
                rewrite_output[:min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None

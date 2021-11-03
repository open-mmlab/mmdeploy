import copy
import importlib
import os
import random
import tempfile

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils.constants import Backend, Codebase
from mmdeploy.utils.test import (WrapModel, get_model_outputs,
                                 get_rewrite_outputs)


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
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
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


@pytest.mark.parametrize('backend_type', ['onnxruntime', 'ncnn', 'openvino'])
def test_anchor_head_get_bboxes(backend_type):
    """Test get_bboxes rewrite of anchor head."""
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
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
            backend_config=dict(type=backend_type),
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


@pytest.mark.parametrize('backend_type', ['onnxruntime', 'ncnn', 'openvino'])
def test_get_bboxes_of_fcos_head(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
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
            backend_config=dict(type=backend_type),
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


def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model


@pytest.mark.parametrize('model_cfg_path', [
    'tests/test_mmdet/data/single_stage_model.json',
    'tests/test_mmdet/data/mask_model.json'
])
@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_forward_of_base_detector_and_visualize(model_cfg_path):
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

    from mmdeploy.apis.utils import visualize
    output_file = tempfile.NamedTemporaryFile(suffix='.jpg').name
    model.CLASSES = [''] * 80
    visualize(
        Codebase.MMDET,
        img.squeeze().permute(1, 2, 0).numpy(),
        result=[torch.rand(0, 5).numpy()] * 80,
        model=model,
        output_file=output_file,
        backend=Backend.ONNXRUNTIME,
        show_result=False)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', ['openvino'])
def test_single_roi_extractor(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')

    single_roi_extractor = get_single_roi_extractor()
    output_names = ['roi_feat']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type),
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


def get_cascade_roi_head(is_with_masks=False):
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
    if is_with_masks:
        args += [mask_roi_extractor, mask_head]

    from mmdet.models import CascadeRoIHead
    model = CascadeRoIHead(*args, **kwargs).eval()
    return model


@pytest.mark.parametrize('backend_type', ['onnxruntime', 'openvino'])
def test_cascade_roi_head(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')

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
    for output in model_outputs[0]:
        if output.shape == (0, 5):
            processed_model_outputs.append(np.zeros((1, 5)))
        else:
            processed_model_outputs.append(output)
    processed_model_outputs = np.array(processed_model_outputs).squeeze()
    processed_model_outputs = processed_model_outputs[None, :, :]

    output_names = ['results']
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type),
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
    processed_backend_outputs = []
    if isinstance(backend_outputs, dict):
        processed_backend_outputs = [
            backend_outputs[name] for name in output_names
            if name in backend_outputs
        ]
    elif isinstance(backend_outputs, (list, tuple)) and \
            backend_outputs[0].shape == (1, 0, 5):
        processed_backend_outputs = np.zeros((1, 80, 5))
    else:
        processed_backend_outputs = backend_outputs
    assert np.allclose(
        processed_model_outputs,
        processed_backend_outputs,
        rtol=1e-03,
        atol=1e-05)


@pytest.mark.parametrize('backend_type', ['openvino'])
def test_cascade_roi_head_with_mask(backend_type):
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')

    cascade_roi_head = get_cascade_roi_head(is_with_masks=True)
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
            backend_config=dict(type=backend_type),
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
    bbox_results = backend_outputs['bbox_results']
    segm_results = backend_outputs['segm_results']
    expected_bbox_results = np.zeros((1, 80, 5))
    expected_segm_results = -np.ones((1, 80))
    assert np.allclose(
        expected_bbox_results, bbox_results, rtol=1e-03,
        atol=1e-05), 'bbox_results do not match.'
    assert np.allclose(
        expected_segm_results, segm_results, rtol=1e-03,
        atol=1e-05), 'segm_results do not match.'

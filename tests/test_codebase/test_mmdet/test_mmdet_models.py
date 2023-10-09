# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
import tempfile
from typing import Dict, List

import mmengine
import numpy as np
import pytest
import torch
from packaging import version

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close

from mmengine import Config
from mmengine.config import ConfigDict

from mmdeploy.codebase import import_codebase
from mmdeploy.core.rewriters.rewriter_manager import RewriterContext
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import (WrapFunction, WrapModel, backend_checker,
                                 check_backend, get_model_outputs,
                                 get_onnx_model, get_rewrite_outputs)

try:
    import_codebase(Codebase.MMDET)
except ImportError:
    pytest.skip(f'{Codebase.MMDET} is not installed.', allow_module_level=True)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('add_ctr_clamp', [True, False])
@pytest.mark.parametrize('clip_border,max_shape',
                         [(False, None), (True, torch.tensor([100, 200]))])
def test_delta2bbox(backend_type: Backend, add_ctr_clamp: bool,
                    clip_border: bool, max_shape: tuple):
    check_backend(backend_type)
    deploy_cfg = Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    # wrap function to enable rewrite
    def delta2bbox(*args, **kwargs):
        import mmdet
        return mmdet.models.task_modules.coders.delta_xywh_bbox_coder. \
            delta2bbox(*args, **kwargs)

    rois = torch.rand(5, 4)
    deltas = torch.rand(5, 4)
    original_outputs = delta2bbox(rois, deltas, add_ctr_clamp=add_ctr_clamp)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(delta2bbox, add_ctr_clamp=add_ctr_clamp)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'rois': rois.unsqueeze(0),
            'deltas': deltas.unsqueeze(0)
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze().cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_tblr2bbox(backend_type: Backend):
    check_backend(backend_type)
    deploy_cfg = Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    # wrap function to enable rewrite
    def tblr2bboxes(*args, **kwargs):
        import mmdet
        return mmdet.models.task_modules.coders.tblr_bbox_coder.tblr2bboxes(
            *args, **kwargs)

    priors = torch.rand(1, 5, 4)
    tblr = torch.rand(1, 5, 4)
    original_outputs = tblr2bboxes(priors, tblr)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(tblr2bboxes)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'priors': priors,
            'tblr': tblr
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test__distancepointbboxcoder__decode(backend_type: Backend):
    check_backend(backend_type)
    deploy_cfg = Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    from mmdet.models.task_modules.coders.distance_point_bbox_coder import \
        DistancePointBBoxCoder
    coder = DistancePointBBoxCoder()
    # wrap function to enable rewrite

    wrapped_model = WrapModel(coder, 'decode')

    points = torch.rand(3, 2)
    pred_bboxes = torch.rand(3, 4)
    original_outputs = coder.decode(points, pred_bboxes)

    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs={
            'points': points,
            'pred_bboxes': pred_bboxes
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize('pre_top_k', [-1, 1000])
def test_multiclass_nms_with_keep_top_k(pre_top_k):
    backend_type = 'onnxruntime'

    from mmdeploy.mmcv.ops import multiclass_nms
    max_output_boxes_per_class = 20
    keep_top_k = 15
    deploy_cfg = Config(
        dict(
            onnx_config=dict(
                output_names=None,
                input_shape=None,
                dynamic_axes=dict(
                    boxes={
                        0: 'batch_size',
                        1: 'num_boxes'
                    },
                    scores={
                        0: 'batch_size',
                        1: 'num_boxes',
                        2: 'num_classes'
                    },
                ),
            ),
            backend_config=dict(type=backend_type),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=max_output_boxes_per_class,
                    pre_top_k=pre_top_k,
                    keep_top_k=keep_top_k,
                    background_label_id=-1,
                ))))

    num_classes = 5
    num_boxes = 2
    batch_size = 1
    export_boxes = torch.rand(batch_size, num_boxes, 4)
    export_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': export_boxes, 'scores': export_scores}

    wrapped_func = WrapFunction(
        multiclass_nms,
        nms_type='nms',
        max_output_boxes_per_class=max_output_boxes_per_class,
        keep_top_k=keep_top_k)

    onnx_model_path = get_onnx_model(
        wrapped_func, model_inputs=model_inputs, deploy_cfg=deploy_cfg)

    num_boxes = 100
    test_boxes = torch.rand(batch_size, num_boxes, 4)
    test_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': test_boxes, 'scores': test_scores}

    import mmdeploy.backend.onnxruntime as ort_apis
    backend_model = ort_apis.ORTWrapper(onnx_model_path, 'cpu', None)
    output = backend_model.forward(model_inputs)
    output = backend_model.output_to_list(output)
    dets = output[0]

    # Subtract 1 dim since we pad the tensors
    assert dets.shape[1] - 1 < keep_top_k, \
        'multiclass_nms returned more values than "keep_top_k"\n' \
        f'dets.shape: {dets.shape}\n' \
        f'keep_top_k: {keep_top_k}'


@backend_checker(Backend.TENSORRT)
def test__anchorgenerator__single_level_grid_priors():
    backend_type = 'tensorrt'
    import onnx
    from mmdet.models.task_modules.prior_generators.anchor_generator import \
        AnchorGenerator

    import mmdeploy.codebase.mmdet.models.task_modules.prior_generators.anchor  # noqa
    from mmdeploy.apis.onnx import export

    generator = AnchorGenerator(
        scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4])

    def single_level_grid_priors(input):
        return generator.single_level_grid_priors(input.shape[2:], 0,
                                                  input.dtype, input.device)

    x = torch.rand(1, 3, 4, 4)
    wrapped_func = WrapFunction(single_level_grid_priors)
    output = wrapped_func(x)

    # test forward
    with RewriterContext({}, backend_type):
        wrap_output = wrapped_func(x)
        torch_assert_close(output, wrap_output)

    onnx_prefix = tempfile.NamedTemporaryFile().name

    export(
        wrapped_func,
        x,
        onnx_prefix,
        backend=backend_type,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dict(input={
            2: 'h',
            3: 'w'
        }))

    onnx_model = onnx.load(onnx_prefix + '.onnx')

    find_trt_grid_priors = False
    for n in onnx_model.graph.node:
        if n.op_type == 'GridPriorsTRT':
            find_trt_grid_priors = True

    assert find_trt_grid_priors


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
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import AnchorHead
    model = AnchorHead(num_classes=4, in_channels=1, test_cfg=test_cfg)
    model.requires_grad_(False)

    return model


def get_ssd_head_model():
    """SSDHead Config."""
    test_cfg = Config(
        dict(
            nms_pre=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200))

    from mmdet.models import SSDHead
    model = SSDHead(
        in_channels=(96, 1280, 512, 256, 256, 128),
        num_classes=4,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        test_cfg=test_cfg)

    model.requires_grad_(False)

    return model


def get_focus_backbone_model():
    """Backbone Focus Config."""
    from mmdet.models.backbones.csp_darknet import Focus
    model = Focus(3, 32)

    model.requires_grad_(False)
    return model


def get_l2norm_forward_model():
    """L2Norm Neck Config."""
    from mmdet.models.necks.ssd_neck import L2Norm
    model = L2Norm(16)
    torch.nn.init.uniform_(model.weight)

    model.requires_grad_(False)
    return model


def get_rpn_head_model():
    """RPN Head Config."""
    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            nms_pre=0,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0))
    from mmdet.models.dense_heads import RPNHead
    model = RPNHead(in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_reppoints_head_model():
    """Reppoints Head Config."""
    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import RepPointsHead
    model = RepPointsHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_detrhead_model():
    """DETR head Config."""
    from mmdet.registry import MODELS
    model = MODELS.build(
        dict(
            type='DETRHead',
            num_classes=4,
            embed_dims=4,
            loss_cls=dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)))
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


def get_gfl_head_model():
    test_cfg = Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))
    anchor_generator = dict(
        type='AnchorGenerator',
        scales_per_octave=1,
        octave_base_scale=8,
        ratios=[1.0],
        strides=[8, 16, 32, 64, 128])
    from mmdet.models.dense_heads import GFLHead
    model = GFLHead(
        num_classes=3,
        in_channels=256,
        reg_max=3,
        test_cfg=test_cfg,
        anchor_generator=anchor_generator)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_focus_forward(backend_type):
    check_backend(backend_type)
    focus_model = get_focus_backbone_model()
    focus_model.cpu().eval()
    s = 128
    seed_everything(1234)
    x = torch.rand(1, 3, s, s)
    model_outputs = [focus_model.forward(x)]
    wrapped_model = WrapModel(focus_model, 'forward')
    rewrite_inputs = {
        'x': x,
    }
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None)))
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs[0], rewrite_outputs):
        model_output = model_output.squeeze()
        rewrite_output = rewrite_output.squeeze()
        torch_assert_close(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_l2norm_forward(backend_type):
    check_backend(backend_type)
    l2norm_neck = get_l2norm_forward_model()
    l2norm_neck.cpu().eval()
    s = 128
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None)))
    seed_everything(1234)
    feat = torch.rand(1, 16, s, s)
    model_outputs = [l2norm_neck.forward(feat)]
    wrapped_model = WrapModel(l2norm_neck, 'forward')
    rewrite_inputs = {
        'x': feat,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        for model_output, rewrite_output in zip(model_outputs[0],
                                                rewrite_outputs[0]):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            assert np.allclose(
                model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        for model_output, rewrite_output in zip(model_outputs[0],
                                                rewrite_outputs[0]):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            assert np.allclose(
                model_output[0], rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_predict_by_feat_of_rpn_head(backend_type: Backend):
    check_backend(backend_type)
    head = get_rpn_head_model()
    head.cpu().eval()
    s = 4
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['dets']
    deploy_cfg = Config(
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
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    # do not run with ncnn backend
    run_with_backend = False if backend_type in [Backend.NCNN] else True
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=run_with_backend)
    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_predict_by_feat_of_gfl_head(backend_type):
    check_backend(backend_type)
    head = get_gfl_head_model()
    head.cpu().eval()
    s = 4
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]
    output_names = ['dets']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                model_type='ncnn_end2end',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    seed_everything(1234)
    cls_score = [
        torch.rand(1, 3, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 16, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    # do not run with ncnn backend
    run_with_backend = False if backend_type in [Backend.NCNN] else True
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=run_with_backend)
    assert rewrite_outputs is not None


def _replace_r50_with_r18(model):
    """Replace ResNet50 with ResNet18 in config."""
    model = copy.deepcopy(model)
    if model.backbone.type == 'ResNet':
        model.backbone.depth = 18
        model.backbone.base_channels = 2
        model.neck.in_channels = [2, 4, 8, 16]
    return model


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('model_cfg_path', [
    'tests/test_codebase/test_mmdet/data/single_stage_model.json',
    'tests/test_codebase/test_mmdet/data/mask_model.json'
])
def test_forward_of_base_detector(model_cfg_path, backend):
    check_backend(backend)
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend.value),
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
                    export_postprocess_mask=False,
                ))))
    model_cfg = Config(dict(model=mmengine.load(model_cfg_path)))
    model_cfg.model = _replace_r50_with_r18(model_cfg.model)
    from mmdet.apis import init_detector
    model = init_detector(model_cfg, None, device='cpu', palette='coco')

    img = torch.randn(1, 3, 64, 64)
    from mmdet.structures import DetDataSample
    data_sample = DetDataSample(metainfo=dict(img_shape=(800, 1216, 3)))
    rewrite_inputs = {'batch_inputs': img}
    wrapped_model = WrapModel(model, 'forward', data_samples=[data_sample])
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.skipif(
    reason='mha only support torch greater than 1.10.0',
    condition=version.parse(torch.__version__) < version.parse('1.10.0'))
@pytest.mark.parametrize(
    'model_cfg_path', ['tests/test_codebase/test_mmdet/data/detr_model.json'])
def test_predict_of_detr_detector(model_cfg_path, backend):
    # Skip test when torch.__version__ < 1.10.0
    # See https://github.com/open-mmlab/mmdeploy/discussions/1434
    check_backend(backend)
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend.value),
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
                    export_postprocess_mask=False,
                ))))
    model_cfg = Config(dict(model=mmengine.load(model_cfg_path)))
    from mmdet.apis import init_detector
    model = init_detector(model_cfg, None, device='cpu', palette='coco')

    img = torch.randn(1, 3, 64, 64)
    from mmdet.structures import DetDataSample
    data_sample = DetDataSample(metainfo=dict(batch_input_shape=(64, 64)))
    rewrite_inputs = {'batch_inputs': img}
    wrapped_model = WrapModel(model, 'forward', data_samples=[data_sample])
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_single_roi_extractor(backend_type: Backend):
    check_backend(backend_type)

    single_roi_extractor = get_single_roi_extractor()
    output_names = ['roi_feat']
    deploy_cfg = Config(
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


def test_single_roi_extractor__ascend():
    check_backend(Backend.ASCEND)

    # create wrap function
    from mmdeploy.utils.test import WrapFunction
    single_roi_extractor = get_single_roi_extractor()
    out_channels = single_roi_extractor.out_channels

    def single_roi_extractor_func(feat0, feat1, feat2, feat3, rois):
        return single_roi_extractor([feat0, feat1, feat2, feat3], rois)

    single_roi_extractor_wrapper = WrapFunction(single_roi_extractor_func)

    # generate data
    seed_everything(1234)
    feats = [
        torch.rand((1, out_channels, 200, 336)),
        torch.rand((1, out_channels, 100, 168)),
        torch.rand((1, out_channels, 50, 84)),
        torch.rand((1, out_channels, 25, 42)),
    ]
    seed_everything(5678)
    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    # create config
    input_names = ['feat0', 'feat1', 'feat2', 'feat3', 'rois']
    output_names = ['roi_feat']
    model_inputs = dict(zip(input_names, feats + [rois]))
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(
                type=Backend.ASCEND.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            feat0=feats[0].shape,
                            feat1=feats[1].shape,
                            feat2=feats[2].shape,
                            feat3=feats[3].shape,
                            rois=rois.shape))
                ]),
            onnx_config=dict(
                input_names=input_names,
                output_names=output_names,
                input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
            )))

    # get torch output
    model_outputs = get_model_outputs(single_roi_extractor_wrapper, 'forward',
                                      model_inputs)

    # get backend output
    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=single_roi_extractor_wrapper,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    if isinstance(backend_outputs, dict):
        backend_outputs = backend_outputs.values()
    for model_output, backend_output in zip(model_outputs[0], backend_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        backend_output = backend_output.squeeze()
        assert model_output.shape == backend_output.shape


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
            'output_size': 1,
            'sampling_ratio': 0
        },
        'out_channels': 4,
        'featmap_strides': [4, 8, 16, 32]
    }
    mask_head = {
        'type': 'FCNMaskHead',
        'num_convs': 4,
        'in_channels': 4,
        'conv_out_channels': 4,
        'num_classes': 80,
        'loss_mask': {
            'type': 'CrossEntropyLoss',
            'use_mask': True,
            'loss_weight': 1.0
        }
    }

    test_cfg = Config(
        dict(
            score_thr=0.05,
            nms=Config(dict(type='nms', iou_threshold=0.5)),
            max_per_img=100,
            mask_thr_binary=0.5))

    args = [num_stages, stage_loss_weights, bbox_roi_extractor, bbox_head]
    kwargs = {'test_cfg': test_cfg}
    if is_instance_seg:
        args += [mask_roi_extractor, mask_head]

    from mmdet.models.roi_heads import CascadeRoIHead
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
    proposals = torch.tensor([[[587.8285, 52.1405, 886.2484, 341.5644, 0.5]]])
    batch_img_metas = {
        'img_shape': torch.tensor([800, 1216]),
        'ori_shape': torch.tensor([800, 1216]),
        'scale_factor': torch.tensor([1, 1, 1, 1])
    }

    output_names = ['results']
    deploy_cfg = Config(
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
    rcnn_test_cfg = ConfigDict(
        dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    model_inputs = {'x': x, 'rpn_results_list': [proposals]}
    wrapped_model = WrapModel(
        cascade_roi_head,
        'predict_bbox',
        batch_img_metas=[batch_img_metas],
        # rpn_results_list=[proposals],
        rcnn_test_cfg=rcnn_test_cfg)
    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)

    assert backend_outputs is not None


def get_fovea_head_model():
    """FoveaHead Config."""
    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import FoveaHead
    model = FoveaHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_predict_by_feat_of_fovea_head(backend_type: Backend):
    check_backend(backend_type)
    fovea_head = get_fovea_head_model()
    fovea_head.cpu().eval()
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
        'batch_img_metas': batch_img_metas
    }
    model_outputs = get_model_outputs(fovea_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        fovea_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes,
                rewrite_outputs[0][i, :, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores,
                rewrite_outputs[0][i, :, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels,
                rewrite_outputs[1][i],
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
        torch.rand((1, 4, 200, 304)),
        torch.rand((1, 4, 100, 152)),
        torch.rand((1, 4, 50, 76)),
        torch.rand((1, 4, 25, 38)),
    ]
    proposals = [
        torch.tensor([[[587.8285, 52.1405, 886.2484, 341.5644, 0.5]]]),
        torch.tensor([[[0]]], dtype=torch.long)
    ]
    batch_img_metas = {
        'img_shape': torch.tensor([800, 1216]),
        'ori_shape': torch.tensor([800, 1216]),
        'scale_factor': torch.tensor([1, 1, 1, 1])
    }

    output_names = ['dets', 'labels', 'masks']
    deploy_cfg = Config(
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
                    export_postprocess_mask=False))))
    model_inputs = {'x': x}
    wrapped_model = WrapModel(
        cascade_roi_head,
        'predict_mask',
        batch_img_metas=[batch_img_metas],
        results_list=proposals)
    backend_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=model_inputs,
        deploy_cfg=deploy_cfg)
    dets = backend_outputs[0]
    labels = backend_outputs[1]
    masks = backend_outputs[2]
    assert dets is not None
    assert labels is not None
    assert masks is not None


def get_yolov3_head_model():
    """yolov3 Head Config."""
    test_cfg = Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=10))
    from mmdet.models.dense_heads import YOLOV3Head
    model = YOLOV3Head(
        num_classes=4,
        in_channels=[16, 8, 4],
        out_channels=[32, 16, 8],
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_yolov3_head_predict_by_feat(backend_type):
    """Test predict_by_feat rewrite of yolov3 head."""
    check_backend(backend_type)
    yolov3_head = get_yolov3_head_model()
    yolov3_head.cpu().eval()
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
    model_inputs = {'pred_maps': pred_maps, 'batch_img_metas': batch_img_metas}
    model_outputs = get_model_outputs(yolov3_head, 'predict_by_feat',
                                      model_inputs)
    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolov3_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'pred_maps': pred_maps,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes,
                rewrite_outputs[0][i, :, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores,
                rewrite_outputs[0][i, :, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels,
                rewrite_outputs[1][i],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def test_yolov3_head_predict_by_feat_ncnn():
    """Test predict_by_feat rewrite of yolov3 head."""
    backend_type = Backend.NCNN
    check_backend(backend_type)
    yolov3_head = get_yolov3_head_model()
    yolov3_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['detection_output']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                model_type='ncnn_end2end',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.45,
                    confidence_threshold=0.005,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                ))))

    seed_everything(1234)
    pred_maps = [
        torch.rand(1, 27, 5, 5),
        torch.rand(1, 27, 10, 10),
        torch.rand(1, 27, 20, 20)
    ]

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolov3_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas[0],
        with_nms=True)
    rewrite_inputs = {
        'pred_maps': pred_maps,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    # output should be of shape [1, N, 6]
    if is_backend_output:
        assert rewrite_outputs[0].shape[-1] == 6
    else:
        assert rewrite_outputs.shape[-1] == 6


def get_centernet_head_model():
    """CenterNet Head Config."""
    test_cfg = Config(dict(topk=100, local_maximum_kernel=3, max_per_img=100))

    from mmdet.models.dense_heads import CenterNetHead
    model = CenterNetHead(8, 8, 4, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_centernet_head_predict_by_feat(backend_type: Backend):
    """Test predict_by_feat rewrite of CenterNetHead."""
    check_backend(backend_type)
    centernet_head = get_centernet_head_model()
    centernet_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'border':
        np.array([11., 99., 11., 99.], dtype=np.float32),
        'img_shape': (s, s),
        'batch_input_shape': (s, s)
    }]

    output_names = ['dets', 'labels']
    deploy_cfg = Config(
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
    center_heatmap_preds = [
        torch.rand(1, centernet_head.num_classes, s // 4, s // 4)
    ]
    seed_everything(5678)
    wh_preds = [torch.rand(1, 2, s // 4, s // 4)]
    seed_everything(9101)
    offset_preds = [torch.rand(1, 2, s // 4, s // 4)]

    # to get outputs of pytorch model
    model_inputs = {
        'center_heatmap_preds': center_heatmap_preds,
        'wh_preds': wh_preds,
        'offset_preds': offset_preds,
        'batch_img_metas': batch_img_metas,
        'with_nms': False
    }
    model_outputs = get_model_outputs(centernet_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        centernet_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'center_heatmap_preds': center_heatmap_preds,
        'wh_preds': wh_preds,
        'offset_preds': offset_preds,
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
            border = batch_img_metas[i]['border']

            rewrite_outputs[0][i, :, 0] -= border[2]
            rewrite_outputs[0][i, :, 1] -= border[0]
            rewrite_outputs[0][i, :, 2] -= border[2]
            rewrite_outputs[0][i, :, 3] -= border[0]
            assert np.allclose(
                model_outputs[i].bboxes[:min_shape],
                rewrite_outputs[0][i, :min_shape, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_yolox_head_model():
    """YOLOX Head Config."""
    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import YOLOXHead
    model = YOLOXHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.ONNXRUNTIME, Backend.OPENVINO])
def test_yolox_head_predict_by_feat(backend_type: Backend):
    """Test predict_by_feat rewrite of YOLOXHead."""
    check_backend(backend_type)
    yolox_head = get_yolox_head_model()
    yolox_head.cpu().eval()
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
        torch.rand(1, yolox_head.num_classes, 2 * pow(2, i), 2 * pow(2, i))
        for i in range(3, 0, -1)
    ]
    seed_everything(5678)
    bbox_preds = [
        torch.rand(1, 4, 2 * pow(2, i), 2 * pow(2, i))
        for i in range(3, 0, -1)
    ]
    seed_everything(9101)
    objectnesses = [
        torch.rand(1, 1, 2 * pow(2, i), 2 * pow(2, i))
        for i in range(3, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
        'batch_img_metas': batch_img_metas,
        'with_nms': True
    }
    model_outputs = get_model_outputs(yolox_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolox_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
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
        # hard code to make two tensors with the same shape
        # rewrite and original codes applied different nms strategy
        min_shape = min(model_outputs[0].bboxes.shape[0],
                        rewrite_outputs[0].shape[1], 5)
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes[:min_shape],
                rewrite_outputs[0][i, :min_shape, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def test_yolox_head_predict_by_feat_ncnn():
    """Test predict_by_feat rewrite of yolox head for ncnn."""
    backend_type = Backend.NCNN
    check_backend(backend_type)
    yolox_head = get_yolox_head_model()
    yolox_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['detection_output']
    deploy_cfg = Config(
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
                    pre_top_k=5000,
                    keep_top_k=10,
                    background_label_id=0,
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

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolox_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    # output should be of shape [1, N, 6]
    if is_backend_output:
        assert rewrite_outputs[0].shape[-1] == 6
    else:
        assert rewrite_outputs.shape[-1] == 6


def get_deploy_cfg(backend_type: Backend, ir_type: str):
    return Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                type=ir_type,
                output_names=['dets', 'labels'],
                input_shape=None),
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


@pytest.mark.parametrize('backend_type, ir_type',
                         [(Backend.ONNXRUNTIME, 'onnx'),
                          (Backend.OPENVINO, 'onnx'),
                          (Backend.TORCHSCRIPT, 'torchscript')])
def test_base_dense_head_predict_by_feat(backend_type: Backend, ir_type: str):
    """Test predict_by_feat rewrite of base dense head."""
    check_backend(backend_type)
    anchor_head = get_anchor_head_model()
    anchor_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    deploy_cfg = get_deploy_cfg(backend_type, ir_type)

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
        'batch_img_metas': batch_img_metas
    }
    model_outputs = get_model_outputs(anchor_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        anchor_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_backend_output:
        # hard code to make two tensors with the same shape
        # rewrite and original codes applied different topk strategy
        # rewrite and original codes applied different nms strategy
        min_shape = min(model_outputs[0].bboxes.shape[0],
                        rewrite_outputs[0].shape[1], 5)
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes[:min_shape],
                rewrite_outputs[0][i, :min_shape, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def test_base_dense_head_predict_by_feat__ncnn():
    """Test predict_by_feat rewrite of base dense head."""
    backend_type = Backend.NCNN
    check_backend(backend_type)
    anchor_head = get_anchor_head_model()
    anchor_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['output']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                model_type='ncnn_end2end',
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

    # to get outputs of onnx model after rewrite
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        anchor_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    # output should be of shape [1, N, 6]
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    assert rewrite_outputs.shape[-1] == 6


@backend_checker(Backend.RKNN)
def test_base_dense_head_get_bboxes__rknn():
    """Test get_bboxes rewrite of ssd head for rknn."""
    ssd_head = get_ssd_head_model()
    ssd_head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]
    output_names = ['output']
    input_names = []
    for i in range(6):
        input_names.append('cls_scores_' + str(i))
        input_names.append('bbox_preds_' + str(i))
    dynamic_axes = None
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=Backend.RKNN.value),
            onnx_config=dict(
                input_names=input_names,
                output_names=output_names,
                input_shape=None,
                dynamic_axes=dynamic_axes),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                model_type='rknn',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                ))))

    # For the ssd_head:
    # the cls_score's size: (1, 30, 20, 20), (1, 30, 10, 10),
    # (1, 30, 5, 5), (1, 30, 3, 3), (1, 30, 2, 2), (1, 30, 1, 1)
    # the bboxes's size: (1, 24, 20, 20), (1, 24, 10, 10),
    # (1, 24, 5, 5), (1, 24, 3, 3), (1, 24, 2, 2), (1, 24, 1, 1)
    feat_shape = [20, 10, 5, 3, 2, 1]
    num_prior = 6
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 30, feat_shape[i], feat_shape[i])
        for i in range(num_prior)
    ]
    seed_everything(5678)
    bboxes = [
        torch.rand(1, 24, feat_shape[i], feat_shape[i])
        for i in range(num_prior)
    ]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = [s, s]
    wrapped_model = WrapModel(
        ssd_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    # output should be of shape [1, N, 4]
    assert rewrite_outputs[0].shape[-1] == 4


@pytest.mark.parametrize('backend_type, ir_type', [(Backend.OPENVINO, 'onnx')])
def test_reppoints_head_predict_by_feat(backend_type: Backend, ir_type: str):
    """Test predict_by_feat rewrite of base dense head."""
    check_backend(backend_type)
    dense_head = get_reppoints_head_model()
    dense_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    deploy_cfg = get_deploy_cfg(backend_type, ir_type)

    # the cls_score's size: (1, 4, 32, 32), (1, 4, 16, 16),
    # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2).
    # the bboxes's size: (1, 4, 32, 32), (1, 4, 16, 16),
    # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2)
    seed_everything(1234)
    cls_score = [
        torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bboxes = [torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'batch_img_metas': batch_img_metas
    }
    model_outputs = get_model_outputs(dense_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    batch_img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        dense_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        # hard code to make two tensors with the same shape
        # rewrite and original codes applied different topk strategy
        # rewrite and original codes applied different nms strategy
        min_shape = min(model_outputs[0].bboxes.shape[0],
                        rewrite_outputs[0].shape[1], 5)
        for i in range(len(model_outputs)):
            assert np.allclose(
                model_outputs[i].bboxes[:min_shape],
                rewrite_outputs[0][i, :min_shape, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend_type, ir_type', [(Backend.OPENVINO, 'onnx')])
def test_reppoints_head_points2bbox(backend_type: Backend, ir_type: str):
    """Test predict_by_feat rewrite of base dense head."""
    check_backend(backend_type)
    dense_head = get_reppoints_head_model()
    dense_head.cpu().eval()
    output_names = ['output']

    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                input_names=['pts'],
                output_names=output_names)))

    # the cls_score's size: (1, 4, 32, 32), (1, 4, 16, 16),
    # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2).
    # the bboxes's size: (1, 4, 32, 32), (1, 4, 16, 16),
    # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2)
    seed_everything(1234)
    pts = torch.rand(1, 18, 16, 16)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(dense_head, 'points2bbox', y_first=True)
    rewrite_inputs = {'pts': pts}
    _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
@pytest.mark.parametrize('backend_type', [(Backend.TENSORRT)])
def test_windows_msa(backend_type: Backend):
    check_backend(backend_type)
    from mmdet.models.backbones.swin import WindowMSA
    model = WindowMSA(96, 3, (7, 7))
    model.cuda().eval()
    output_names = ['output']

    deploy_cfg = Config(
        dict(
            backend_config=dict(
                type=backend_type.value,
                common_config=dict(fp16_mode=True, max_workspace_size=1 << 20),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            x=dict(
                                min_shape=[12, 49, 96],
                                opt_shape=[12, 49, 96],
                                max_shape=[12, 49, 96]),
                            mask=dict(
                                min_shape=[12, 49, 49],
                                opt_shape=[12, 49, 49],
                                max_shape=[12, 49, 49])))
                ]),
            onnx_config=dict(
                input_shape=None,
                input_names=['x', 'mask'],
                output_names=output_names)))

    x = torch.randn([12, 49, 96]).cuda()
    mask = torch.randn([12, 49, 49]).cuda()
    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'x': x, 'mask': mask}
    _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
@pytest.mark.parametrize('backend_type', [(Backend.TENSORRT)])
def test_shift_windows_msa(backend_type: Backend):
    check_backend(backend_type)
    from mmdet.models.backbones.swin import ShiftWindowMSA
    model = ShiftWindowMSA(96, 3, 7)
    model.cuda().eval()
    output_names = ['output']

    deploy_cfg = Config(
        dict(
            backend_config=dict(
                type=backend_type.value,
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            query=dict(
                                min_shape=[1, 60800, 96],
                                opt_shape=[1, 60800, 96],
                                max_shape=[1, 60800, 96])))
                ]),
            onnx_config=dict(
                input_shape=None,
                input_names=['query'],
                output_names=output_names)))

    query = torch.randn([1, 60800, 96]).cuda()
    hw_shape = (torch.tensor(200), torch.tensor(304))

    wrapped_model = WrapModel(model, 'forward')
    rewrite_inputs = {'query': query, 'hw_shape': hw_shape}
    _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
@pytest.mark.parametrize('backend_type', [(Backend.TENSORRT)])
def test_mlvl_point_generator__single_level_grid_priors__tensorrt(
        backend_type: Backend):
    check_backend(backend_type)
    from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
    model = MlvlPointGenerator([8, 16, 32])
    output_names = ['output']

    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                input_names=['featmap_size', 'level_idx'],
                output_names=output_names)))

    featmap_size = torch.tensor([80, 80])
    with_stride = True

    wrapped_model = WrapModel(
        model, 'single_level_grid_priors', with_stride=with_stride)
    rewrite_inputs = {
        'featmap_size': featmap_size,
        'level_idx': torch.tensor(0)
    }
    _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)


@pytest.mark.parametrize('backend_type, ir_type',
                         [(Backend.ONNXRUNTIME, 'onnx')])
def test_detrhead__predict_by_feat(backend_type: Backend, ir_type: str):
    """Test predict_by_feat rewrite of detr head."""
    check_backend(backend_type)
    dense_head = get_detrhead_model()
    dense_head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    deploy_cfg = get_deploy_cfg(backend_type, ir_type)

    seed_everything(1234)
    cls_score = [torch.rand(1, 100, 5) for i in range(5, 0, -1)]
    seed_everything(5678)
    bboxes = [torch.rand(1, 100, 4) for i in range(5, 0, -1)]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        dense_head, 'predict_by_feat', batch_img_metas=img_metas)
    rewrite_inputs = {
        'all_cls_scores_list': cls_score,
        'all_bbox_preds_list': bboxes,
    }
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert rewrite_outputs is not None


def get_solo_head_model():
    test_cfg = Config(
        dict(
            nms_pre=500,
            score_thr=0.1,
            mask_thr=0.5,
            filter_thr=0.05,
            kernel='gaussian',  # gaussian/linear
            sigma=2.0,
            max_per_img=100))
    from mmdet.models.dense_heads import SOLOHead
    model = SOLOHead(4, 32, feat_channels=32, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.OPENVINO])
def test_solo_head_predict_by_feat(backend_type: Backend):
    """Test predict_by_feat rewrite of solo head."""
    check_backend(backend_type)
    solo_head = get_solo_head_model()
    s = 128
    solo_head.cpu().eval()
    batch_img_metas = [{'img_shape': (s, s, 3), 'ori_shape': (s, s, 3)}]

    output_names = ['dets', 'labels', 'masks']
    deploy_cfg = Config(
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
                    export_postprocess_mask=True))))
    seed_everything(1234)
    num_grids = [24, 20, 16, 12, 8]
    mask_preds = [
        torch.rand(1, num_grid**2, s // 4, s // 4) for num_grid in num_grids
    ]
    seed_everything(5678)
    cls_scores = [
        torch.rand(1, solo_head.num_classes, num_grid, num_grid)
        for num_grid in num_grids
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'mlvl_mask_preds': mask_preds,
        'mlvl_cls_scores': cls_scores,
        'batch_img_metas': batch_img_metas,
    }
    model_outputs = get_model_outputs(solo_head, 'predict_by_feat',
                                      model_inputs)

    wrapped_model = WrapModel(
        solo_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'mlvl_mask_preds': mask_preds,
        'mlvl_cls_scores': cls_scores,
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
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_rtmdet_head_model():

    from mmdet.models.dense_heads import RTMDetHead
    from mmdet.models.task_modules.prior_generators.point_generator import \
        MlvlPointGenerator

    test_cfg = Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))
    model = RTMDetHead(1, 64)
    model.prior_generator = MlvlPointGenerator([8, 4, 2])
    model.test_cfg = test_cfg

    model.requires_grad_(False)
    return model


def test_rtmdet_head_predict_by_feat_ncnn():
    """Test predict_by_feat rewrite of yolov3 head."""
    backend_type = Backend.NCNN
    check_backend(backend_type)
    rtmdet_head = get_rtmdet_head_model()
    rtmdet_head.cpu().eval()
    s = 320
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['detection_output']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                model_type='ncnn_end2end',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.45,
                    confidence_threshold=0.005,
                    max_output_boxes_per_class=200,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                ))))

    seed_everything(1234)
    cls_scores = [
        torch.rand(1, 1, 40, 40),
        torch.rand(1, 1, 20, 20),
        torch.rand(1, 1, 10, 10)
    ]

    bbox_preds = [
        torch.rand(1, 4, 40, 40),
        torch.rand(1, 4, 20, 20),
        torch.rand(1, 4, 10, 10)
    ]

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        rtmdet_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {'cls_scores': cls_scores, 'bbox_preds': bbox_preds}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    # output should be of shape [1, N, 6]
    if is_backend_output:
        assert rewrite_outputs[0].shape[-1] == 6
    else:
        assert rewrite_outputs.shape[-1] == 6


def get_solov2_head_model():
    """solov2 Head Config."""
    test_cfg = Config(
        dict(
            nms_pre=500,
            score_thr=0.1,
            mask_thr=0.5,
            filter_thr=0.05,
            kernel='gaussian',  # gaussian/linear
            sigma=2.0,
            max_per_img=100))
    mask_feature_head_cfg = Config(
        dict(
            feat_channels=16,
            start_level=0,
            end_level=3,
            out_channels=32,
            mask_stride=4,
            norm_cfg=dict(
                type='GN',
                num_groups=4,
            )))
    from mmdet.models.dense_heads import SOLOV2Head
    model = SOLOV2Head(
        num_classes=4,
        in_channels=32,
        feat_channels=64,
        mask_feature_head=mask_feature_head_cfg,
        test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.OPENVINO])
def test_solov2_head_predict_by_feat(backend_type):
    """Test predict_by_feat rewrite of solov2 head."""
    check_backend(backend_type)
    solov2_head = get_solov2_head_model()
    solov2_head.cpu().eval()
    s = 128
    batch_img_metas = [{'img_shape': (s, s, 3), 'ori_shape': (s, s, 3)}]

    output_names = ['dets', 'labels', 'masks']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.45,
                    max_output_boxes_per_class=20,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                    export_postprocess_mask=True))))

    seed_everything(1234)
    num_grids = [24, 20, 16, 12, 8]
    kernel_preds = [
        torch.rand(1, solov2_head.mask_feature_head.out_channels, num_grid,
                   num_grid) for num_grid in num_grids
    ]
    cls_scores = [
        torch.rand(1, solov2_head.num_classes, num_grid, num_grid)
        for num_grid in num_grids
    ]
    mask_feats = torch.rand(1, solov2_head.mask_feature_head.out_channels,
                            s // 4, s // 4)
    # to get outputs of pytorch model
    model_inputs = {
        'mlvl_kernel_preds': kernel_preds,
        'mlvl_cls_scores': cls_scores,
        'mask_feats': mask_feats,
        'batch_img_metas': batch_img_metas,
    }
    model_outputs = get_model_outputs(solov2_head, 'predict_by_feat',
                                      model_inputs)
    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        solov2_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'mlvl_kernel_preds': kernel_preds,
        'mlvl_cls_scores': cls_scores,
        'mask_feats': mask_feats,
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
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None


def get_condinst_bbox_head():
    """condinst Bbox Head Config."""
    test_cfg = Config(
        dict(
            mask_thr=0.5,
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.05))
    from mmdet.models.dense_heads import CondInstBboxHead
    model = CondInstBboxHead(
        center_sampling=True,
        centerness_on_reg=True,
        conv_bias=True,
        dcn_on_last_conv=False,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes=80,
        num_params=169,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        test_cfg=test_cfg,
    )

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_condinst_bbox_head_predict_by_feat(backend_type):
    """Test predict_by_feat rewrite of condinst bbox head."""
    check_backend(backend_type)
    condinst_bbox_head = get_condinst_bbox_head()
    condinst_bbox_head.cpu().eval()
    s = 128
    batch_img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['dets', 'labels', 'param_preds', 'points', 'strides']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    confidence_threshold=0.005,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                    export_postprocess_mask=False))))

    seed_everything(1234)
    cls_scores = [
        torch.rand(1, condinst_bbox_head.num_classes, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]
    seed_everything(5678)
    bbox_preds = [
        torch.rand(1, 4, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(9101)
    score_factors = [
        torch.rand(1, 1, pow(2, i), pow(2, i)) for i in range(5, 0, -1)
    ]
    seed_everything(1121)
    param_preds = [
        torch.rand(1, condinst_bbox_head.num_params, pow(2, i), pow(2, i))
        for i in range(5, 0, -1)
    ]

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        condinst_bbox_head, 'predict_by_feat', batch_img_metas=batch_img_metas)
    rewrite_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'score_factors': score_factors,
        'param_preds': param_preds,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        dets = rewrite_outputs[0]
        labels = rewrite_outputs[1]
        param_preds = rewrite_outputs[2]
        points = rewrite_outputs[3]
        strides = rewrite_outputs[4]
        assert dets.shape[-1] == 5
        assert labels is not None
        assert param_preds.shape[-1] == condinst_bbox_head.num_params
        assert points.shape[-1] == 2
        assert strides is not None
    else:
        assert rewrite_outputs is not None


def get_condinst_mask_head():
    """condinst Mask Head Config."""
    test_cfg = Config(
        dict(
            mask_thr=0.5,
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.05))
    from mmdet.models.dense_heads import CondInstMaskHead
    model = CondInstMaskHead(
        mask_feature_head=dict(
            end_level=2,
            feat_channels=128,
            in_channels=256,
            mask_stride=8,
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_stacked_convs=4,
            out_channels=8,
            start_level=0),
        num_layers=3,
        feat_channels=8,
        mask_out_stride=4,
        size_of_interest=8,
        max_masks_to_train=300,
        loss_mask=dict(
            activate=True,
            eps=5e-06,
            loss_weight=1.0,
            type='DiceLoss',
            use_sigmoid=True),
        test_cfg=test_cfg,
    )

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_condinst_mask_head_forward(backend_type):
    """Test predict_by_feat rewrite of condinst mask head."""
    check_backend(backend_type)

    output_names = ['mask_preds']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    class TestCondInstMaskHeadModel(torch.nn.Module):

        def __init__(self, condinst_mask_head):
            super(TestCondInstMaskHeadModel, self).__init__()
            self.mask_head = condinst_mask_head

        def forward(self, x, param_preds, points, strides):
            positive_infos = dict(
                param_preds=param_preds, points=points, strides=strides)
            return self.mask_head(x, positive_infos)

    mask_head = get_condinst_mask_head()
    level = mask_head.mask_feature_head.end_level - \
        mask_head.mask_feature_head.start_level + 1

    condinst_mask_head = TestCondInstMaskHeadModel(mask_head)
    condinst_mask_head.cpu().eval()

    seed_everything(1234)
    x = [torch.rand(1, 256, pow(2, i), pow(2, i)) for i in range(level, 0, -1)]
    seed_everything(5678)
    param_preds = torch.rand(1, 100, 169)
    seed_everything(9101)
    points = torch.rand(1, 100, 2)
    seed_everything(1121)
    strides = torch.rand(1, 100)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(condinst_mask_head, 'forward')
    rewrite_inputs = {
        'x': x,
        'param_preds': param_preds,
        'points': points,
        'strides': strides
    }
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None

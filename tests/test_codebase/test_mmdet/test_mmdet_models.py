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
from mmdeploy.utils.config_utils import get_ir_config
from mmdeploy.utils.test import (WrapModel, backend_checker, check_backend,
                                 get_model_outputs, get_rewrite_outputs)

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close


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


def get_head_inputs(seed, channels, num_inputs):
    """Generate inputs for the head."""
    seed_everything(seed)
    return [
        torch.rand(1, channels, pow(2, i), pow(2, i))
        for i in range(num_inputs, 0, -1)
    ]


@pytest.fixture
def fcos_head_model():
    """FCOS Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import FCOSHead
    model = FCOSHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.fixture
def focus_backbone_model():
    """Backbone Focus Config."""
    from mmdet.models.backbones.csp_darknet import Focus
    model = Focus(3, 32)

    model.requires_grad_(False)
    return model


@pytest.fixture
def l2norm_forward_model():
    """L2Norm Neck Config."""
    from mmdet.models.necks.ssd_neck import L2Norm
    model = L2Norm(16)
    torch.nn.init.uniform_(model.weight)

    model.requires_grad_(False)
    return model


@pytest.fixture
def rpn_head_model():
    """RPN Head Config."""
    test_cfg = mmcv.Config(
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


@pytest.fixture
def detrhead_model():
    """DETR head Config."""
    from mmdet.models import build_head
    model = build_head(
        dict(
            type='DETRHead',
            num_classes=4,
            in_channels=1,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=4,
                                num_heads=1,
                                dropout=0.1)
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=4,
                            feedforward_channels=32,
                            num_fcs=2,
                            ffn_drop=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        feedforward_channels=32,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=1,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=4,
                            num_heads=1,
                            dropout=0.1),
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=4,
                            feedforward_channels=32,
                            num_fcs=2,
                            ffn_drop=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        feedforward_channels=32,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=2, normalize=True),
            test_cfg=dict(max_per_img=100)))
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_focus_forward(backend_type, focus_backbone_model):
    check_backend(backend_type)
    focus_model = focus_backbone_model
    focus_model.cpu().eval()
    s = 128
    seed_everything(1234)
    x = torch.rand(1, 3, s, s)
    model_outputs = [focus_model.forward(x)]
    wrapped_model = WrapModel(focus_model, 'forward')
    rewrite_inputs = {
        'x': x,
    }
    deploy_cfg = mmcv.Config(
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
def test_l2norm_forward(backend_type, l2norm_forward_model):
    check_backend(backend_type)
    l2norm_neck = l2norm_forward_model
    l2norm_neck.cpu().eval()
    s = 128
    deploy_cfg = mmcv.Config(
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


def test_get_bboxes_of_fcos_head_ncnn(fcos_head_model):
    backend_type = Backend.NCNN
    check_backend(backend_type)
    fcos_head = fcos_head_model
    fcos_head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    output_names = ['detection_output']
    deploy_cfg = mmcv.Config(
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
    cls_score = get_head_inputs(1234, fcos_head.num_classes, 5)
    bboxes = get_head_inputs(5678, 4, 5)
    centernesses = get_head_inputs(9101, 1, 5)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        fcos_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'centernesses': centernesses
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


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_get_bboxes_of_rpn_head(backend_type: Backend, rpn_head_model):
    check_backend(backend_type)
    head = rpn_head_model
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
    cls_score = get_head_inputs(1234, 9, 5)
    bboxes = get_head_inputs(5678, 36, 5)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        head, 'get_bboxes', img_metas=img_metas, with_nms=True)
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


class TestGFLHead:

    @pytest.fixture
    def gfl_head(self):
        test_cfg = mmcv.Config(
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
        model.cpu().eval()
        return model

    @pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
    def test_get_bboxes_of_gfl_head(self, backend_type, gfl_head):
        check_backend(backend_type)
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
                    model_type='ncnn_end2end',
                    post_processing=dict(
                        score_threshold=0.05,
                        iou_threshold=0.5,
                        max_output_boxes_per_class=200,
                        pre_top_k=5000,
                        keep_top_k=100,
                        background_label_id=-1,
                    ))))

        cls_score = get_head_inputs(1234, 3, 5)
        bboxes = get_head_inputs(5678, 16, 5)

        # to get outputs of onnx model after rewrite
        img_metas[0]['img_shape'] = torch.Tensor([s, s])
        wrapped_model = WrapModel(
            gfl_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
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
    def test_forward_of_gfl_head(self, backend_type, gfl_head):
        check_backend(backend_type)
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(input_shape=None)))
        feats = get_head_inputs(1234, 256, 5)
        model_outputs = [gfl_head.forward(feats)]
        wrapped_model = WrapModel(gfl_head, 'forward')
        rewrite_inputs = {
            'feats': feats,
        }
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)
        model_outputs[0] = [*model_outputs[0][0], *model_outputs[0][1]]
        for model_output, rewrite_output in zip(model_outputs[0],
                                                rewrite_outputs[0]):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            assert np.allclose(
                model_output, rewrite_output, rtol=1e-03, atol=1e-05)


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
    deploy_cfg = mmcv.Config(
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


class TestSingleRoIExtractor:

    @pytest.fixture(scope='class')
    def single_roi_extractor(self):
        """SingleRoIExtractor Config."""
        from mmdet.models.roi_heads import SingleRoIExtractor
        roi_layer = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
        out_channels = 1
        featmap_strides = [4, 8, 16, 32]
        model = SingleRoIExtractor(roi_layer, out_channels,
                                   featmap_strides).eval()

        return model

    @pytest.fixture(scope='class')
    def feats(self, single_roi_extractor):
        seed_everything(1234)
        out_channels = single_roi_extractor.out_channels
        return [
            torch.rand((1, out_channels, 200, 336)),
            torch.rand((1, out_channels, 100, 168)),
            torch.rand((1, out_channels, 50, 84)),
            torch.rand((1, out_channels, 25, 42)),
        ]

    @pytest.fixture(scope='class')
    def rois(self):
        seed_everything(5678)
        return torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    @pytest.mark.parametrize('backend_type',
                             [Backend.ONNXRUNTIME, Backend.OPENVINO])
    def test_single_roi_extractor(self, backend_type: Backend,
                                  single_roi_extractor, feats, rois):
        check_backend(backend_type)

        output_names = ['roi_feat']
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=backend_type.value),
                onnx_config=dict(output_names=output_names, input_shape=None),
                codebase_config=dict(
                    type='mmdet',
                    task='ObjectDetection',
                )))

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
        for model_output, backend_output in zip(model_outputs[0],
                                                backend_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            backend_output = backend_output.squeeze()
            assert np.allclose(
                model_output, backend_output, rtol=1e-03, atol=1e-05)

    def test_single_roi_extractor__ascend(self, single_roi_extractor, feats,
                                          rois):
        check_backend(Backend.ASCEND)

        # create wrap function
        from mmdeploy.utils.test import WrapFunction

        def single_roi_extractor_func(feat0, feat1, feat2, feat3, rois):
            return single_roi_extractor([feat0, feat1, feat2, feat3], rois)

        single_roi_extractor_wrapper = WrapFunction(single_roi_extractor_func)

        # create config
        input_names = ['feat0', 'feat1', 'feat2', 'feat3', 'rois']
        output_names = ['roi_feat']
        model_inputs = dict(zip(input_names, feats + [rois]))
        deploy_cfg = mmcv.Config(
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
        model_outputs = get_model_outputs(single_roi_extractor_wrapper,
                                          'forward', model_inputs)

        # get backend output
        backend_outputs, _ = get_rewrite_outputs(
            wrapped_model=single_roi_extractor_wrapper,
            model_inputs=model_inputs,
            deploy_cfg=deploy_cfg)
        if isinstance(backend_outputs, dict):
            backend_outputs = backend_outputs.values()
        for model_output, backend_output in zip(model_outputs[0],
                                                backend_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            backend_output = backend_output.squeeze()
            assert model_output.shape == backend_output.shape


class TestCascadeRoIHead:

    @pytest.fixture(scope='class')
    def cascade_roi_head(self, request):
        """CascadeRoIHead Config."""
        is_instance_seg = request.node.get_closest_marker(
            'is_instance_seg', False)
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

        from mmdet.models.roi_heads import CascadeRoIHead
        model = CascadeRoIHead(*args, **kwargs).eval()
        return model

    @pytest.fixture(scope='class')
    def model_inputs(self):
        seed_everything(1234)
        x = [
            torch.rand((1, 64, 200, 304)),
            torch.rand((1, 64, 100, 152)),
            torch.rand((1, 64, 50, 76)),
            torch.rand((1, 64, 25, 38)),
        ]
        proposals = torch.tensor([[587.8285, 52.1405, 886.2484, 341.5644,
                                   0.5]])
        return {'x': x, 'proposals': proposals.unsqueeze(0)}

    @pytest.fixture(scope='class')
    def img_metas(self):
        return {
            'img_shape': torch.tensor([800, 1216]),
            'ori_shape': torch.tensor([800, 1216]),
            'scale_factor': torch.tensor([1, 1, 1, 1])
        }

    @pytest.fixture(scope='class')
    def wrapped_model(self, cascade_roi_head, img_metas):
        return WrapModel(
            cascade_roi_head, 'simple_test', img_metas=[img_metas])

    @pytest.mark.parametrize('backend_type',
                             [Backend.ONNXRUNTIME, Backend.OPENVINO])
    def test_cascade_roi_head(self, backend_type: Backend, model_inputs,
                              wrapped_model):
        check_backend(backend_type)

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
        backend_outputs, _ = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=model_inputs,
            deploy_cfg=deploy_cfg)

        assert backend_outputs is not None

    @pytest.mark.parametrize('backend_type', [Backend.OPENVINO])
    @pytest.mark.is_instance_seg(True)
    def test_cascade_roi_head_with_mask(self, backend_type: Backend,
                                        model_inputs, wrapped_model):
        check_backend(backend_type)

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
        backend_outputs, _ = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=model_inputs,
            deploy_cfg=deploy_cfg)
        bbox_results = backend_outputs[0]
        segm_results = backend_outputs[1]
        assert bbox_results is not None
        assert segm_results is not None


@pytest.fixture
def fovea_head_model():
    """FoveaHead Config."""
    test_cfg = mmcv.Config(
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
def test_get_bboxes_of_fovea_head(backend_type: Backend, fovea_head_model):
    check_backend(backend_type)
    fovea_head = fovea_head_model
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
    cls_score = get_head_inputs(1234, fovea_head.num_classes, 5)
    bboxes = get_head_inputs(5678, 4, 5)

    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(fovea_head, 'get_bboxes', model_inputs)

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(fovea_head, 'get_bboxes', img_metas=img_metas)
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


class TestYOLOV3Head:

    @pytest.fixture(scope='class')
    def yolov3_head(self):
        """yolov3 Head Config."""
        test_cfg = mmcv.Config(
            dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                conf_thr=0.005,
                nms=dict(type='nms', iou_threshold=0.45),
                max_per_img=100))
        from mmdet.models.dense_heads import YOLOV3Head
        model = YOLOV3Head(
            num_classes=4,
            in_channels=[16, 8, 4],
            out_channels=[32, 16, 8],
            test_cfg=test_cfg)

        model.requires_grad_(False)
        model.cpu().eval()
        return model

    @pytest.fixture(scope='class')
    def img_metas(self):
        s = 128
        return [{
            'scale_factor': np.ones(4),
            'pad_shape': (s, s, 3),
            'img_shape': (s, s, 3)
        }]

    @pytest.fixture(scope='class')
    def pred_maps(self):
        seed_everything(1234)
        return [
            torch.rand(1, 27, 5, 5),
            torch.rand(1, 27, 10, 10),
            torch.rand(1, 27, 20, 20)
        ]

    @pytest.mark.parametrize('backend_type',
                             [Backend.ONNXRUNTIME, Backend.OPENVINO])
    def test_yolov3_head_get_bboxes(self, backend_type, yolov3_head, img_metas,
                                    pred_maps):
        """Test get_bboxes rewrite of yolov3 head."""
        check_backend(backend_type)

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

        # to get outputs of pytorch model
        model_inputs = {'pred_maps': pred_maps, 'img_metas': img_metas}
        model_outputs = get_model_outputs(yolov3_head, 'get_bboxes',
                                          model_inputs)

        # to get outputs of onnx model after rewrite
        wrapped_model = WrapModel(
            yolov3_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
        rewrite_inputs = {
            'pred_maps': pred_maps,
        }
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)

        if is_backend_output:
            if isinstance(rewrite_outputs, dict):
                rewrite_outputs = convert_to_list(rewrite_outputs,
                                                  output_names)
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

    def test_yolov3_head_get_bboxes_ncnn(self, yolov3_head, img_metas,
                                         pred_maps):
        """Test get_bboxes rewrite of yolov3 head."""
        backend_type = Backend.NCNN
        check_backend(backend_type)

        output_names = ['detection_output']
        deploy_cfg = mmcv.Config(
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
        # output should be of shape [1, N, 6]
        if is_backend_output:
            assert rewrite_outputs[0].shape[-1] == 6
        else:
            assert rewrite_outputs.shape[-1] == 6


class TestYOLOXHead:

    @pytest.fixture(scope='class')
    def yolox_head(self):
        """YOLOX Head Config."""
        test_cfg = mmcv.Config(
            dict(
                deploy_nms_pre=0,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))

        from mmdet.models.dense_heads import YOLOXHead
        model = YOLOXHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

        model.requires_grad_(False)
        model.cpu().eval()
        return model

    @pytest.fixture(scope='class')
    def img_metas(self):
        s = 128
        return [{
            'scale_factor': np.ones(4),
            'pad_shape': (s, s, 3),
            'img_shape': (s, s, 3)
        }]

    @pytest.fixture(scope='class')
    def wrapped_model(self, yolox_head, img_metas):
        return WrapModel(
            yolox_head, 'get_bboxes', img_metas=img_metas, with_nms=True)

    @pytest.fixture(scope='class')
    def rewrite_inputs(self, yolox_head):

        cls_scores = get_head_inputs(1234, yolox_head.num_classes, 3)
        bbox_preds = get_head_inputs(5678, 4, 3)
        objectnesses = get_head_inputs(9101, 1, 3)

        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'objectnesses': objectnesses
        }

    @pytest.fixture(scope='class')
    def model_outputs(self, yolox_head, rewrite_inputs, img_metas):
        model_inputs = {'img_metas': img_metas}
        model_inputs.update(rewrite_inputs)
        return get_model_outputs(yolox_head, 'get_bboxes', model_inputs)

    @pytest.mark.parametrize('backend_type',
                             [Backend.ONNXRUNTIME, Backend.OPENVINO])
    def test_yolox_head_get_bboxes(self, backend_type: Backend, rewrite_inputs,
                                   model_outputs, wrapped_model):
        """Test get_bboxes rewrite of YOLOXHead."""
        check_backend(backend_type)
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

        # to get outputs of onnx model after rewrite
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)

        if is_backend_output:
            if isinstance(rewrite_outputs, dict):
                rewrite_outputs = convert_to_list(rewrite_outputs,
                                                  output_names)
            for model_output, rewrite_output in zip(model_outputs[0],
                                                    rewrite_outputs):
                model_output = model_output.squeeze().cpu().numpy()
                rewrite_output = rewrite_output.squeeze().cpu().numpy()
                # hard code to make two tensors with the same shape
                # rewrite and original codes applied different nms strategy
                min_shape = min(model_output.shape[0], rewrite_output.shape[0],
                                5)
                assert np.allclose(
                    model_output[:min_shape],
                    rewrite_output[:min_shape],
                    rtol=1e-03,
                    atol=1e-05)
        else:
            assert rewrite_outputs is not None

    def test_yolox_head_get_bboxes_ncnn(self, rewrite_inputs, wrapped_model):
        """Test get_bboxes rewrite of yolox head for ncnn."""
        backend_type = Backend.NCNN
        check_backend(backend_type)

        output_names = ['detection_output']
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
                        pre_top_k=5000,
                        keep_top_k=10,
                        background_label_id=0,
                    ))))

        # to get outputs of onnx model after rewrite
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)
        # output should be of shape [1, N, 6]
        if is_backend_output:
            assert rewrite_outputs[0].shape[-1] == 6
        else:
            assert rewrite_outputs.shape[-1] == 6


@pytest.fixture
def vfnet_head_model():
    """VFNet Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
    from mmdet.models.dense_heads import VFNetHead
    model = VFNetHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    model.cpu().eval()
    return model


@pytest.mark.parametrize('backend_type',
                         [Backend.OPENVINO, Backend.ONNXRUNTIME])
def test_get_bboxes_of_vfnet_head(backend_type: Backend, vfnet_head_model):
    """Test get_bboxes rewrite of VFNet head."""
    check_backend(backend_type)
    vfnet_head = vfnet_head_model
    vfnet_head.cpu().eval()
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

    cls_score = get_head_inputs(1234, vfnet_head.num_classes, 5)
    bboxes = get_head_inputs(5678, 4, 5)

    model_inputs = {
        'cls_scores': cls_score,
        'bbox_preds': bboxes,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(vfnet_head, 'get_bboxes', model_inputs)

    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(
        vfnet_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
    rewrite_inputs = {'cls_scores': cls_score, 'bbox_preds': bboxes}
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


def get_deploy_cfg(backend_type: Backend, ir_type: str):
    return mmcv.Config(
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


class TestBaseDenseHead:

    @pytest.fixture(scope='class')
    def anchor_head(self):
        """AnchorHead Config."""
        test_cfg = mmcv.Config(
            dict(
                deploy_nms_pre=0,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))

        from mmdet.models.dense_heads import AnchorHead
        model = AnchorHead(num_classes=4, in_channels=1, test_cfg=test_cfg)
        model.requires_grad_(False)

        model.cpu().eval()
        return model

    @pytest.fixture(scope='class')
    def rewrite_inputs(self):
        # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
        # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
        cls_score = get_head_inputs(1234, 36, 5)
        bboxes = get_head_inputs(5678, 36, 5)
        return {
            'cls_scores': cls_score,
            'bbox_preds': bboxes,
        }

    @pytest.fixture(scope='class')
    def img_metas(self):
        s = 128
        img_metas = [{
            'scale_factor': np.ones(4),
            'pad_shape': (s, s, 3),
            'img_shape': (s, s, 3)
        }]
        img_metas[0]['img_shape'] = torch.tensor([s, s])

        return img_metas

    @pytest.fixture(scope='class')
    def wrapped_model(self, anchor_head, img_metas):
        return WrapModel(
            anchor_head, 'get_bboxes', img_metas=img_metas, with_nms=True)

    @pytest.fixture(scope='class')
    def model_outputs(self, rewrite_inputs, anchor_head, img_metas):
        model_inputs = {'img_metas': img_metas}
        model_inputs.update(rewrite_inputs)
        return get_model_outputs(anchor_head, 'get_bboxes', model_inputs)

    @pytest.mark.parametrize('backend_type, ir_type',
                             [(Backend.ONNXRUNTIME, 'onnx'),
                              (Backend.OPENVINO, 'onnx'),
                              (Backend.TORCHSCRIPT, 'torchscript')])
    def test_base_dense_head_get_bboxes(self, backend_type: Backend,
                                        ir_type: str, rewrite_inputs,
                                        wrapped_model, model_outputs):
        """Test get_bboxes rewrite of base dense head."""
        check_backend(backend_type)
        deploy_cfg = get_deploy_cfg(backend_type, ir_type)
        output_names = get_ir_config(deploy_cfg).get('output_names', None)

        # to get outputs of onnx model after rewrite
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)

        if is_backend_output:
            if isinstance(rewrite_outputs, dict):
                rewrite_outputs = convert_to_list(rewrite_outputs,
                                                  output_names)
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

    def test_base_dense_head_get_bboxes__ncnn(self, rewrite_inputs,
                                              wrapped_model):
        """Test get_bboxes rewrite of base dense head."""
        backend_type = Backend.NCNN
        check_backend(backend_type)

        output_names = ['output']
        deploy_cfg = mmcv.Config(
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

        # to get outputs of onnx model after rewrite
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)

        # output should be of shape [1, N, 6]
        if is_backend_output:
            rewrite_outputs = rewrite_outputs[0]

        assert rewrite_outputs.shape[-1] == 6


class TestSSDHead:

    feat_shape = [20, 10, 5, 3, 2, 1]
    num_prior = 6

    @pytest.fixture(scope='class')
    def ssd_head(self):
        """SSDHead Config."""
        test_cfg = mmcv.Config(
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

        model.cpu().eval()
        return model

    @pytest.fixture(scope='class')
    def rewrite_inputs(self):
        # the cls_score's size: (1, 30, 20, 20), (1, 30, 10, 10),
        # (1, 30, 5, 5), (1, 30, 3, 3), (1, 30, 2, 2), (1, 30, 1, 1)
        seed_everything(1234)
        cls_scores = [
            torch.rand(1, 30, self.feat_shape[i], self.feat_shape[i])
            for i in range(self.num_prior)
        ]
        seed_everything(5678)
        bbox_preds = [
            torch.rand(1, 24, self.feat_shape[i], self.feat_shape[i])
            for i in range(self.num_prior)
        ]

        return dict(cls_scores=cls_scores, bbox_preds=bbox_preds)

    @pytest.fixture(scope='class')
    def img_metas(self):
        s = 128
        img_metas = [{
            'scale_factor': np.ones(4),
            'pad_shape': (s, s, 3),
            'img_shape': (s, s, 3)
        }]

        return img_metas

    @pytest.mark.parametrize('is_dynamic', [True, False])
    def test_ssd_head_get_bboxes__ncnn(self, is_dynamic: bool, ssd_head,
                                       rewrite_inputs, img_metas):
        """Test get_bboxes rewrite of ssd head for ncnn."""
        check_backend(Backend.NCNN)
        output_names = ['output']
        input_names = []
        for i in range(6):
            input_names.append('cls_scores_' + str(i))
            input_names.append('bbox_preds_' + str(i))
        dynamic_axes = None
        if is_dynamic:
            dynamic_axes = {
                output_names[0]: {
                    1: 'num_dets',
                }
            }
            for input_name in input_names:
                dynamic_axes[input_name] = {2: 'height', 3: 'width'}
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type=Backend.NCNN.value),
                onnx_config=dict(
                    input_names=input_names,
                    output_names=output_names,
                    input_shape=None,
                    dynamic_axes=dynamic_axes),
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

        img_metas = copy.deepcopy(img_metas)
        s = 128
        img_metas[0]['img_shape'] = torch.tensor([s, s
                                                  ]) if is_dynamic else [s, s]
        wrapped_model = WrapModel(
            ssd_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg)

        # output should be of shape [1, N, 6]
        if is_backend_output:
            rewrite_outputs = rewrite_outputs[0]

        assert rewrite_outputs.shape[-1] == 6

    @backend_checker(Backend.RKNN)
    def test_base_dense_head_get_bboxes__rknn(self, ssd_head, rewrite_inputs,
                                              img_metas):
        """Test get_bboxes rewrite of ssd head for rknn."""
        output_names = ['output']
        input_names = []
        for i in range(6):
            input_names.append('cls_scores_' + str(i))
            input_names.append('bbox_preds_' + str(i))
        dynamic_axes = None
        deploy_cfg = mmcv.Config(
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

        # to get outputs of onnx model after rewrite
        img_metas = copy.deepcopy(img_metas)
        s = 128
        img_metas[0]['img_shape'] = [s, s]
        wrapped_model = WrapModel(
            ssd_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_model=wrapped_model,
            model_inputs=rewrite_inputs,
            deploy_cfg=deploy_cfg,
            run_with_backend=False)

        # output should be of shape [1, N, 4]
        assert rewrite_outputs[0].shape[-1] == 4


class TestReppointsHead:

    @pytest.fixture(scope='class')
    def dense_head(self):
        """Reppoints Head Config."""
        test_cfg = mmcv.Config(
            dict(
                deploy_nms_pre=0,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))

        from mmdet.models.dense_heads import RepPointsHead
        model = RepPointsHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

        model.requires_grad_(False)
        model.cpu().eval()
        return model

    @pytest.mark.parametrize('backend_type, ir_type',
                             [(Backend.OPENVINO, 'onnx')])
    def test_reppoints_head_get_bboxes(self, backend_type: Backend,
                                       ir_type: str, dense_head):
        """Test get_bboxes rewrite of base dense head."""
        check_backend(backend_type)
        s = 128
        img_metas = [{
            'scale_factor': np.ones(4),
            'pad_shape': (s, s, 3),
            'img_shape': (s, s, 3)
        }]

        deploy_cfg = get_deploy_cfg(backend_type, ir_type)
        output_names = get_ir_config(deploy_cfg).get('output_names', None)

        # the cls_score's size: (1, 4, 32, 32), (1, 4, 16, 16),
        # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2).
        # the bboxes's size: (1, 4, 32, 32), (1, 4, 16, 16),
        # (1, 4, 8, 8), (1, 4, 4, 4), (1, 4, 2, 2)
        cls_score = get_head_inputs(1234, 4, 5)
        bboxes = get_head_inputs(5678, 4, 5)

        # to get outputs of pytorch model
        model_inputs = {
            'cls_scores': cls_score,
            'bbox_preds': bboxes,
            'img_metas': img_metas
        }
        model_outputs = get_model_outputs(dense_head, 'get_bboxes',
                                          model_inputs)

        # to get outputs of onnx model after rewrite
        img_metas[0]['img_shape'] = torch.Tensor([s, s])
        wrapped_model = WrapModel(
            dense_head, 'get_bboxes', img_metas=img_metas, with_nms=True)
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
                rewrite_outputs = convert_to_list(rewrite_outputs,
                                                  output_names)
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

    @pytest.mark.parametrize('backend_type', [(Backend.OPENVINO)])
    def test_reppoints_head_points2bbox(self, backend_type: Backend,
                                        dense_head):
        """Test get_bboxes rewrite of base dense head."""
        check_backend(backend_type)
        output_names = ['output']

        deploy_cfg = mmcv.Config(
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

    deploy_cfg = mmcv.Config(
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

    deploy_cfg = mmcv.Config(
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
    from mmdet.core.anchor import MlvlPointGenerator
    model = MlvlPointGenerator([8, 16, 32])
    output_names = ['output']

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(
                input_shape=None,
                input_names=['query'],
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
def test_detrhead_get_bboxes(backend_type: Backend, ir_type: str,
                             detrhead_model):
    """Test get_bboxes rewrite of base dense head."""
    check_backend(backend_type)
    dense_head = detrhead_model
    dense_head.cpu().eval()
    s = 128
    img_metas = [{
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3)
    }]

    deploy_cfg = get_deploy_cfg(backend_type, ir_type)

    seed_everything(1234)
    cls_score = [[torch.rand(1, 100, 5) for i in range(5, 0, -1)]]
    seed_everything(5678)
    bboxes = [[torch.rand(1, 100, 4) for i in range(5, 0, -1)]]

    # to get outputs of onnx model after rewrite
    img_metas[0]['img_shape'] = torch.Tensor([s, s])
    wrapped_model = WrapModel(dense_head, 'get_bboxes', img_metas=img_metas)
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

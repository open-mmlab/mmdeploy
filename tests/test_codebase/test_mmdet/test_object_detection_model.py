# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import pytest
import torch
from mmengine import Config
from mmengine.structures import BaseDataElement, InstanceData

import mmdeploy.backend.ncnn as ncnn_apis
import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMDET)
except ImportError:
    pytest.skip(f'{Codebase.MMDET} is not installed.', allow_module_level=True)

from mmdeploy.codebase.mmdet.deploy.object_detection_model import End2EndModel


def assert_det_results(results, module_name: str = 'model'):
    assert results is not None, f'failed to get output using {module_name}'
    assert isinstance(results, Sequence)
    assert len(results) == 2
    assert results[0].shape[0] == results[1].shape[0]
    assert results[0].shape[1] == results[1].shape[1]


def assert_forward_results(results, module_name: str = 'model'):
    assert results is not None, f'failed to get output using {module_name}'
    assert isinstance(results, Sequence)
    assert len(results) == 1
    assert isinstance(results[0].pred_instances, InstanceData)
    assert results[0].pred_instances.bboxes.shape[-1] == 4
    assert results[0].pred_instances.scores.shape[0] == \
        results[0].pred_instances.labels.shape[0] == \
        results[0].pred_instances.bboxes.shape[0]


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'dets': torch.rand(1, 10, 5),
            'labels': torch.rand(1, 10)
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config(
            {'onnx_config': {
                'output_names': ['dets', 'labels']
            }})

        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu',
                                         deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        imgs = torch.rand(1, 3, 64, 64)
        img_metas = [
            BaseDataElement(metainfo={
                'img_shape': [64, 64],
                'scale_factor': [1, 1]
            })
        ]
        results = self.end2end_model.forward(imgs, img_metas)
        assert_forward_results(results, 'End2EndModel')

    def test_predict(self):
        imgs = torch.rand(1, 3, 64, 64)
        dets, labels = self.end2end_model.predict(imgs)
        assert dets.shape[-1] == 5
        assert labels.shape[0] == dets.shape[0]


@backend_checker(Backend.ONNXRUNTIME)
class TestMaskEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        num_classes = 80
        num_dets = 10
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'dets': torch.rand(1, num_dets, 5),
            'labels': torch.randint(num_classes, (1, num_dets)),
            'masks': torch.rand(1, num_dets, 28, 28)
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({
            'onnx_config': {
                'output_names': ['dets', 'labels', 'masks']
            },
            'codebase_config': {
                'post_processing': {
                    'export_postprocess_mask': False
                }
            }
        })

        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu',
                                         deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        imgs = torch.rand(1, 3, 64, 64)
        img_metas = [
            BaseDataElement(
                metainfo={
                    'img_shape': [64, 64],
                    'ori_shape': [32, 32],
                    'scale_factor': [1, 1]
                })
        ]
        results = self.end2end_model.forward(imgs, img_metas)
        assert_forward_results(results, 'mask End2EndModel')


def get_test_cfg_and_post_processing():
    test_cfg = {
        'nms_pre': 100,
        'min_bbox_size': 0,
        'score_thr': 0.05,
        'nms': {
            'type': 'nms',
            'iou_threshold': 0.5
        },
        'max_per_img': 10
    }
    post_processing = {
        'score_threshold': 0.05,
        'iou_threshold': 0.5,
        'max_output_boxes_per_class': 20,
        'pre_top_k': -1,
        'keep_top_k': 10,
        'background_label_id': -1
    }
    return test_cfg, post_processing


def prepare_model_deploy_cfgs():
    test_cfg, post_processing = get_test_cfg_and_post_processing()
    bbox_roi_extractor = {
        'type': 'SingleRoIExtractor',
        'roi_layer': {
            'type': 'RoIAlign',
            'output_size': 7,
            'sampling_ratio': 0
        },
        'out_channels': 8,
        'featmap_strides': [4]
    }
    bbox_head = {
        'type': 'Shared2FCBBoxHead',
        'in_channels': 8,
        'fc_out_channels': 1024,
        'roi_feat_size': 7,
        'num_classes': 80,
        'bbox_coder': {
            'type': 'DeltaXYWHBBoxCoder',
            'target_means': [0.0, 0.0, 0.0, 0.0],
            'target_stds': [0.1, 0.1, 0.2, 0.2]
        },
        'reg_class_agnostic': False,
        'loss_cls': {
            'type': 'CrossEntropyLoss',
            'use_sigmoid': False,
            'loss_weight': 1.0
        },
        'loss_bbox': {
            'type': 'L1Loss',
            'loss_weight': 1.0
        }
    }
    roi_head = dict(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head)
    model_cfg = Config(
        dict(
            model=dict(
                neck=dict(num_outs=0),
                test_cfg=dict(rpn=test_cfg, rcnn=test_cfg),
                roi_head=roi_head)))
    deploy_cfg = Config(
        dict(codebase_config=dict(post_processing=post_processing)))
    return model_cfg, deploy_cfg


class DummyWrapper(torch.nn.Module):

    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, *arg, **kwargs):
        return 0

    def output_to_list(self, *arg, **kwargs):
        return self.outputs


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize('partition_type', [None, 'end2end'])
def test_build_object_detection_model(partition_type):
    _, post_processing = get_test_cfg_and_post_processing()
    model_cfg = Config(dict(data=dict(test={'type': 'CocoDataset'})))
    deploy_cfg = Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['dets', 'labels']),
            codebase_config=dict(
                type='mmdet', post_processing=post_processing)))
    if partition_type:
        deploy_cfg.partition_config = dict(
            apply_marks=True,
            type=partition_type,
            partition_cfg=[dict(output_names=[])])

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            build_object_detection_model
        detector = build_object_detection_model([''], model_cfg, deploy_cfg,
                                                'cpu')
        assert isinstance(detector, End2EndModel)


@backend_checker(Backend.NCNN)
class TestNCNNEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.ncnn import NCNNWrapper
        ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(NCNNWrapper)
        cls.outputs = {
            'output': torch.rand(1, 10, 6),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({'onnx_config': {'output_names': ['output']}})
        model_cfg = Config({})

        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            NCNNEnd2EndModel
        cls.ncnn_end2end_model = NCNNEnd2EndModel(Backend.NCNN, ['', ''],
                                                  'cpu', model_cfg, deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    @pytest.mark.parametrize('num_det', [10, 0])
    def test_predict(self, num_det):
        self.outputs = {
            'output': torch.rand(1, num_det, 6),
        }
        imgs = torch.rand(1, 3, 64, 64)
        results = self.ncnn_end2end_model.predict(imgs)
        assert_det_results(results, 'NCNNEnd2EndModel')


@backend_checker(Backend.RKNN)
class TestRKNNModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        import mmdeploy.backend.rknn as rknn_apis
        from mmdeploy.backend.rknn import RKNNWrapper
        rknn_apis.__dict__.update({'RKNNWrapper': RKNNWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(RKNNWrapper)
        cls.outputs = [
            torch.rand(1, 255, 5, 5),
            torch.rand(1, 255, 10, 10),
            torch.rand(1, 255, 20, 20)
        ]
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({
            'onnx_config': {
                'output_names': ['output']
            },
            'backend_config': {
                'common_config': {}
            }
        })
        model_cfg = Config(
            dict(
                model=dict(
                    bbox_head=dict(
                        type='YOLOV3Head',
                        num_classes=80,
                        in_channels=[512, 256, 128],
                        out_channels=[1024, 512, 256],
                        anchor_generator=dict(
                            type='YOLOAnchorGenerator',
                            base_sizes=[[(116, 90), (156, 198), (
                                373, 326)], [(30, 61), (62, 45), (
                                    59, 119)], [(10, 13), (16, 30), (33, 23)]],
                            strides=[32, 16, 8]),
                        bbox_coder=dict(type='YOLOBBoxCoder'),
                        featmap_strides=[32, 16, 8],
                        loss_cls=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=True,
                            loss_weight=1.0,
                            reduction='sum'),
                        loss_conf=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=True,
                            loss_weight=1.0,
                            reduction='sum'),
                        loss_xy=dict(
                            type='CrossEntropyLoss',
                            use_sigmoid=True,
                            loss_weight=2.0,
                            reduction='sum'),
                        loss_wh=dict(
                            type='MSELoss', loss_weight=2.0, reduction='sum')),
                    test_cfg=dict(
                        nms_pre=1000,
                        min_bbox_size=0,
                        score_thr=0.05,
                        conf_thr=0.005,
                        nms=dict(type='nms', iou_threshold=0.45),
                        max_per_img=100))))

        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            RKNNModel
        cls.rknn_model = RKNNModel(Backend.RKNN, ['', ''], 'cpu',
                                   ['' for i in range(80)], model_cfg,
                                   deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward_test(self):
        imgs = torch.rand(1, 3, 64, 64)
        results = self.rknn_model.forward_test(imgs)
        assert_det_results(results, 'RKNNWrapper')

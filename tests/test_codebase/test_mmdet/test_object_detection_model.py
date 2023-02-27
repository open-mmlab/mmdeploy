# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase.mmdet.deploy.object_detection_model import End2EndModel
from mmdeploy.utils import Backend
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker


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
    if isinstance(results[0], tuple):  # mask
        assert len(results[0][0]) == 80
    else:
        assert len(results[0]) == 80


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'dets': torch.rand(1, 10, 5),
                'labels': torch.rand(1, 10)
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config(
                {'onnx_config': {
                    'output_names': ['dets', 'labels']
                }})

            from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
                End2EndModel
            model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu',
                                 ['' for i in range(80)], deploy_cfg)
            yield model

    def test_forward(self, end2end_model):
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [1, 1, 1, 1],
            'border': [0, 0, 0]
        }]]
        results = end2end_model.forward(imgs, img_metas)
        assert_forward_results(results, 'End2EndModel')

    def test_show_result(self, end2end_model, tmp_path):
        input_img = np.zeros([64, 64, 3])
        img_path = str(tmp_path / 'tmp.jpg')

        result = (torch.rand(1, 10, 5), torch.rand(1, 10))
        end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path)


@backend_checker(Backend.ONNXRUNTIME)
class TestMaskEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        num_classes = 80
        num_dets = 10
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'dets': torch.rand(1, num_dets, 5),
                'labels': torch.randint(num_classes, (1, num_dets)),
                'masks': torch.rand(1, num_dets, 28, 28)
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
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
            model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu',
                                 ['' for i in range(80)], deploy_cfg)
            yield model

    def test_forward(self, end2end_model):
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [1, 1, 1, 1],
        }]]
        results = end2end_model.forward(imgs, img_metas)
        assert_forward_results(results, 'mask End2EndModel')


@pytest.fixture(scope='module')
def cfg_and_post_processing():
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


@backend_checker(Backend.ONNXRUNTIME)
class TestPartitionSingleStageModel:

    @pytest.fixture(scope='class')
    def model(self, cfg_and_post_processing):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'scores': torch.rand(1, 10, 80),
                'boxes': torch.rand(1, 10, 4)
            }
            wrapper.set(outputs=outputs)

            test_cfg, post_processing = cfg_and_post_processing
            model_cfg = mmcv.Config(dict(model=dict(test_cfg=test_cfg)))
            deploy_cfg = mmcv.Config(
                dict(codebase_config=dict(post_processing=post_processing)))

            from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
                PartitionSingleStageModel
            model_ = PartitionSingleStageModel(
                Backend.ONNXRUNTIME, [''],
                'cpu', ['' for i in range(80)],
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg)
            yield model_

    def test_forward_test(self, model):
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [1, 1, 1, 1],
        }]]
        results = model.forward_test(imgs, img_metas)
        assert_det_results(results, 'PartitionSingleStageModel')

    def test_postprocess(self, model):
        scores = torch.rand(1, 120, 80)
        bboxes = torch.rand(1, 120, 4)

        results = model.partition0_postprocess(scores=scores, bboxes=bboxes)
        assert_det_results(
            results, '.partition0_postprocess of'
            'PartitionSingleStageModel')


@pytest.fixture(scope='module')
def model_deploy_cfgs(cfg_and_post_processing):
    test_cfg, post_processing = cfg_and_post_processing
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
    model_cfg = mmcv.Config(
        dict(
            model=dict(
                neck=dict(num_outs=0),
                test_cfg=dict(rpn=test_cfg, rcnn=test_cfg),
                roi_head=roi_head)))
    deploy_cfg = mmcv.Config(
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
class TestPartitionTwoStageModel:

    @pytest.fixture(scope='class')
    def model(self, model_deploy_cfgs):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = [
                np.random.rand(1, 12, 80).astype(np.float32),
                np.random.rand(1, 12, 4).astype(np.float32),
            ] * 2

            model_cfg, deploy_cfg = model_deploy_cfgs

            wrapper.set(
                outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

            # replace original function in PartitionTwoStageModel
            from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
                PartitionTwoStageModel

            model_ = PartitionTwoStageModel(
                Backend.ONNXRUNTIME, ['', ''],
                'cpu', ['' for i in range(80)],
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg)
            feats = [torch.randn(1, 8, 14, 14) for i in range(5)]
            scores = torch.rand(1, 10, 1)
            bboxes = torch.rand(1, 10, 4)
            bboxes[..., 2:4] = 2 * bboxes[..., :2]

            cls_score = torch.rand(10, 81)
            bbox_pred = torch.rand(10, 320)

            model_.device = 'cpu'
            model_.CLASSES = ['' for i in range(80)]
            model_.first_wrapper = DummyWrapper([*feats, scores, bboxes])
            model_.second_wrapper = DummyWrapper([cls_score, bbox_pred])

            yield model_

    def test_postprocess(self, model):
        feats = [torch.randn(1, 8, 14, 14) for i in range(5)]
        scores = torch.rand(1, 50, 1)
        bboxes = torch.rand(1, 50, 4)
        bboxes[..., 2:4] = 2 * bboxes[..., :2]

        results = model.partition0_postprocess(
            x=feats, scores=scores, bboxes=bboxes)
        assert results is not None, 'failed to get output using '\
            'partition0_postprocess of PartitionTwoStageDetector'
        assert isinstance(results, tuple)
        assert len(results) == 2

        rois = torch.rand(1, 10, 5)
        cls_score = torch.rand(10, 81)
        bbox_pred = torch.rand(10, 320)
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [1, 1, 1, 1],
        }]]
        results = model.partition1_postprocess(
            rois=rois,
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            img_metas=img_metas)
        assert results is not None, 'failed to get output using '\
            'partition1_postprocess of PartitionTwoStageDetector'
        assert isinstance(results, tuple)
        assert len(results) == 2

    def test_forward(self, model):

        class DummyPTSDetector(torch.nn.Module):
            """A dummy wrapper for unit tests."""

            def __init__(self, *args, **kwargs):
                self.output_names = ['dets', 'labels']

            def partition0_postprocess(self, *args, **kwargs):
                return self.outputs0

            def partition1_postprocess(self, *args, **kwargs):
                return self.outputs1

        import types
        model.partition0_postprocess = types.MethodType(
            DummyPTSDetector.partition0_postprocess, model)
        model.partition1_postprocess = types.MethodType(
            DummyPTSDetector.partition1_postprocess, model)
        model.outputs0 = [torch.rand(2, 3)] * 2
        model.outputs1 = [torch.rand(1, 9, 5), torch.rand(1, 9)]

        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [1, 1, 1, 1],
        }]]
        results = model.forward(imgs, img_metas)
        assert_forward_results(results, 'PartitionTwoStageModel')


class TestGetClassesFromCfg:
    data_cfg1 = mmcv.Config(
        dict(
            data=dict(
                test=dict(type='CocoDataset'),
                val=dict(type='CityscapesDataset'),
                train=dict(type='CityscapesDataset'))))

    data_cfg2 = mmcv.Config(
        dict(
            data=dict(
                val=dict(type='CocoDataset'),
                train=dict(type='CityscapesDataset'))))
    data_cfg3 = mmcv.Config(dict(data=dict(train=dict(type='CocoDataset'))))
    data_cfg4 = mmcv.Config(dict(data=dict(error=dict(type='CocoDataset'))))

    data_cfg_classes_1 = mmcv.Config(
        dict(
            data=dict(
                test=dict(classes=('a')),
                val=dict(classes=('b')),
                train=dict(classes=('b')))))

    data_cfg_classes_2 = mmcv.Config(
        dict(data=dict(val=dict(classes=('a')), train=dict(classes=('b')))))
    data_cfg_classes_3 = mmcv.Config(
        dict(data=dict(train=dict(classes=('a')))))
    data_cfg_classes_4 = mmcv.Config(dict(classes=('a')))

    @pytest.mark.parametrize('cfg',
                             [data_cfg1, data_cfg2, data_cfg3, data_cfg4])
    def test_get_classes_from_cfg(self, cfg):
        from mmdet.datasets import DATASETS

        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            get_classes_from_config

        if 'error' in cfg.data:
            with pytest.raises(RuntimeError):
                get_classes_from_config(cfg)
        else:
            assert get_classes_from_config(
                cfg) == DATASETS.module_dict['CocoDataset'].CLASSES

    @pytest.mark.parametrize('cfg', [
        data_cfg_classes_1, data_cfg_classes_2, data_cfg_classes_3,
        data_cfg_classes_4
    ])
    def test_get_classes_from_custom_cfg(self, cfg):
        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            get_classes_from_config

        assert get_classes_from_config(cfg) == ['a']


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize('partition_type', [None, 'end2end'])
def test_build_object_detection_model(partition_type, cfg_and_post_processing):
    _, post_processing = cfg_and_post_processing
    model_cfg = mmcv.Config(dict(data=dict(test={'type': 'CocoDataset'})))
    deploy_cfg = mmcv.Config(
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

    @pytest.fixture(scope='class', params=[10, 0])
    def end2end_model(self, request):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.ncnn import NCNNWrapper

        # simplify backend inference
        with SwitchBackendWrapper(NCNNWrapper) as wrapper:
            param = request.param
            outputs = {
                'output': torch.rand(1, param, 6),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config(
                {'onnx_config': {
                    'output_names': ['output']
                }})
            model_cfg = mmcv.Config({})

            from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
                NCNNEnd2EndModel
            ncnn_end2end_model = NCNNEnd2EndModel(Backend.NCNN, ['', ''],
                                                  'cpu',
                                                  ['' for i in range(80)],
                                                  model_cfg, deploy_cfg)
            yield ncnn_end2end_model

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(1, 3, 64, 64)
        results = end2end_model.forward_test(imgs)
        assert_det_results(results, 'NCNNEnd2EndModel')


@backend_checker(Backend.RKNN)
class TestRKNNModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.rknn import RKNNWrapper

        # simplify backend inference
        with SwitchBackendWrapper(RKNNWrapper) as wrapper:
            outputs = [
                torch.rand(1, 255, 5, 5),
                torch.rand(1, 255, 10, 10),
                torch.rand(1, 255, 20, 20)
            ]
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
                'onnx_config': {
                    'output_names': ['output']
                },
                'backend_config': {
                    'common_config': {}
                }
            })
            model_cfg = mmcv.Config(
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
                                        59,
                                        119)], [(10, 13), (16, 30), (33, 23)]],
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
                                type='MSELoss',
                                loss_weight=2.0,
                                reduction='sum')),
                        test_cfg=dict(
                            nms_pre=1000,
                            min_bbox_size=0,
                            score_thr=0.05,
                            conf_thr=0.005,
                            nms=dict(type='nms', iou_threshold=0.45),
                            max_per_img=100))))

            from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
                RKNNModel
            rknn_model = RKNNModel(Backend.RKNN, ['', ''], 'cpu',
                                   ['' for i in range(80)], model_cfg,
                                   deploy_cfg)
            return rknn_model

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(1, 3, 64, 64)
        results = end2end_model.forward_test(imgs)
        assert_det_results(results, 'RKNNWrapper')

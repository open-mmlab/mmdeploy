import importlib

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.apis.ncnn as ncnn_apis
import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.ppl as ppl_apis
import mmdeploy.apis.tensorrt as trt_apis
from mmdeploy.utils.test import SwitchBackendWrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTDetector():
    # force add backend wrapper regardless of plugins
    # make sure TensorRTDetector can use TRTWrapper inside itself
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'dets': torch.rand(1, 100, 5).cuda(),
        'labels': torch.rand(1, 100).cuda()
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmdet.apis.inference import TensorRTDetector
        trt_detector = TensorRTDetector('', ['' for i in range(80)], 0)
        imgs = [torch.rand(1, 3, 64, 64).cuda()]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = trt_detector.forward(imgs, img_metas)
        assert results is not None, ('failed to get output using '
                                     'TensorRTDetector')


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeDetector():
    # force add backend wrapper regardless of plugins
    # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = (torch.rand(1, 100, 5), torch.rand(1, 100))
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmdet.apis.inference import ONNXRuntimeDetector
        ort_detector = ONNXRuntimeDetector('', ['' for i in range(80)], 0)
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ort_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeDetector'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLDetector():
    # force add backend wrapper regardless of plugins
    # make sure PPLDetector can use PPLWrapper inside itself
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    # simplify backend inference
    outputs = (torch.rand(1, 100, 5), torch.rand(1, 100))
    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmdet.apis.inference import PPLDetector
        ppl_detector = PPLDetector('', ['' for i in range(80)], 0)
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ppl_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using PPLDetector'


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


def test_PartitionSingleStageDetector():
    test_cfg, post_processing = get_test_cfg_and_post_processing()
    model_cfg = mmcv.Config(dict(model=dict(test_cfg=test_cfg)))
    deploy_cfg = mmcv.Config(
        dict(codebase_config=dict(post_processing=post_processing)))

    from mmdeploy.mmdet.apis.inference import PartitionSingleStageDetector
    pss_detector = PartitionSingleStageDetector(['' for i in range(80)],
                                                model_cfg=model_cfg,
                                                deploy_cfg=deploy_cfg,
                                                device_id=0)
    scores = torch.rand(1, 120, 80)
    bboxes = torch.rand(1, 120, 4)

    results = pss_detector.partition0_postprocess(scores=scores, bboxes=bboxes)
    assert results is not None, 'failed to get output using '\
        'partition0_postprocess of PartitionSingleStageDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNPSSDetector():
    test_cfg, post_processing = get_test_cfg_and_post_processing()
    model_cfg = mmcv.Config(dict(model=dict(test_cfg=test_cfg)))
    deploy_cfg = mmcv.Config(
        dict(codebase_config=dict(post_processing=post_processing)))

    # force add backend wrapper regardless of plugins
    # make sure NCNNPSSDetector can use NCNNWrapper inside itself
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    # simplify backend inference
    outputs = {
        'scores': torch.rand(1, 120, 80),
        'boxes': torch.rand(1, 120, 4)
    }
    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        from mmdeploy.mmdet.apis.inference import NCNNPSSDetector

        ncnn_pss_detector = NCNNPSSDetector(['', ''], ['' for i in range(80)],
                                            model_cfg=model_cfg,
                                            deploy_cfg=deploy_cfg,
                                            device_id=0)
        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ncnn_pss_detector.forward(imgs, img_metas)
        assert results is not None, ('failed to get output using '
                                     'NCNNPSSDetector')


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimePSSDetector():
    test_cfg, post_processing = get_test_cfg_and_post_processing()
    model_cfg = mmcv.Config(dict(model=dict(test_cfg=test_cfg)))
    deploy_cfg = mmcv.Config(
        dict(codebase_config=dict(post_processing=post_processing)))

    # force add backend wrapper regardless of plugins
    # make sure ONNXRuntimePSSDetector can use ORTWrapper inside itself
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = [
        np.random.rand(1, 120, 80).astype(np.float32),
        np.random.rand(1, 120, 4).astype(np.float32)
    ]
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        from mmdeploy.mmdet.apis.inference import ONNXRuntimePSSDetector

        ort_pss_detector = ONNXRuntimePSSDetector(
            '', ['' for i in range(80)],
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            device_id=0)
        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ort_pss_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'ONNXRuntimePSSDetector'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTPSSDetector():
    test_cfg, post_processing = get_test_cfg_and_post_processing()
    model_cfg = mmcv.Config(dict(model=dict(test_cfg=test_cfg)))
    deploy_cfg = mmcv.Config(
        dict(codebase_config=dict(post_processing=post_processing)))

    # force add backend wrapper regardless of plugins
    # make sure TensorRTPSSDetector can use TRTWrapper inside itself
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'scores': torch.rand(1, 120, 80).cuda(),
        'boxes': torch.rand(1, 120, 4).cuda()
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        from mmdeploy.mmdet.apis.inference import TensorRTPSSDetector

        trt_pss_detector = TensorRTPSSDetector(
            '', ['' for i in range(80)],
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            device_id=0)
        imgs = [torch.rand(1, 3, 32, 32).cuda()]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = trt_pss_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'TensorRTPSSDetector'


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
    model_cfg = mmcv.Config(
        dict(
            model=dict(
                test_cfg=dict(rpn=test_cfg, rcnn=test_cfg),
                roi_head=roi_head)))
    deploy_cfg = mmcv.Config(
        dict(codebase_config=dict(post_processing=post_processing)))
    return model_cfg, deploy_cfg


def test_PartitionTwoStageDetector():
    model_cfg, deploy_cfg = prepare_model_deploy_cfgs()
    from mmdeploy.mmdet.apis.inference import PartitionTwoStageDetector
    pts_detector = PartitionTwoStageDetector(['' for i in range(80)],
                                             model_cfg=model_cfg,
                                             deploy_cfg=deploy_cfg,
                                             device_id=0)
    feats = [torch.randn(1, 8, 14, 14) for i in range(5)]
    scores = torch.rand(1, 50, 1)
    bboxes = torch.rand(1, 50, 4)
    bboxes[..., 2:4] = 2 * bboxes[..., :2]
    results = pts_detector.partition0_postprocess(
        x=feats, scores=scores, bboxes=bboxes)
    assert results is not None, 'failed to get output using '\
        'partition0_postprocess of PartitionTwoStageDetector'

    rois = torch.rand(1, 10, 5)
    cls_score = torch.rand(10, 81)
    bbox_pred = torch.rand(10, 320)
    img_metas = [[{
        'ori_shape': [32, 32, 3],
        'img_shape': [32, 32, 3],
        'scale_factor': [2.09, 1.87, 2.09, 1.87],
    }]]
    results = pts_detector.partition1_postprocess(
        rois=rois,
        cls_score=cls_score,
        bbox_pred=bbox_pred,
        img_metas=img_metas)
    assert results is not None, 'failed to get output using '\
        'partition1_postprocess of PartitionTwoStageDetector'


class DummyPTSDetector(torch.nn.Module):
    """A dummy wrapper for unit tests."""

    def __init__(self, *args, **kwargs):
        self.output_names = ['dets', 'labels']

    def partition0_postprocess(self, *args, **kwargs):
        return self.outputs0

    def partition1_postprocess(self, *args, **kwargs):
        return self.outputs1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTPTSDetector():
    model_cfg, deploy_cfg = prepare_model_deploy_cfgs()

    # force add backend wrapper regardless of plugins
    # make sure TensorRTPTSDetector can use TRTWrapper inside itself
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'scores': torch.rand(1, 12, 80).cuda(),
        'boxes': torch.rand(1, 12, 4).cuda(),
        'cls_score': torch.rand(1, 12, 80).cuda(),
        'bbox_pred': torch.rand(1, 12, 4).cuda()
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        # replace original function in PartitionTwoStageDetector
        from mmdeploy.mmdet.apis.inference import PartitionTwoStageDetector
        PartitionTwoStageDetector.__init__ = DummyPTSDetector.__init__
        PartitionTwoStageDetector.partition0_postprocess = \
            DummyPTSDetector.partition0_postprocess
        PartitionTwoStageDetector.partition1_postprocess = \
            DummyPTSDetector.partition1_postprocess
        PartitionTwoStageDetector.outputs0 = [torch.rand(2, 3).cuda()] * 2
        PartitionTwoStageDetector.outputs1 = [
            torch.rand(1, 9, 5).cuda(),
            torch.rand(1, 9).cuda()
        ]
        PartitionTwoStageDetector.device_id = 0
        PartitionTwoStageDetector.CLASSES = ['' for i in range(80)]

        from mmdeploy.mmdet.apis.inference import TensorRTPTSDetector
        trt_pts_detector = TensorRTPTSDetector(['', ''],
                                               ['' for i in range(80)],
                                               model_cfg=model_cfg,
                                               deploy_cfg=deploy_cfg,
                                               device_id=0)

        imgs = [torch.rand(1, 3, 32, 32).cuda()]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]
        results = trt_pts_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'TensorRTPTSDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimePTSDetector():
    model_cfg, deploy_cfg = prepare_model_deploy_cfgs()

    # force add backend wrapper regardless of plugins
    # make sure ONNXRuntimePTSDetector can use TRTWrapper inside itself
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = [
        np.random.rand(1, 12, 80).astype(np.float32),
        np.random.rand(1, 12, 4).astype(np.float32),
    ] * 2
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        # replace original function in PartitionTwoStageDetector
        from mmdeploy.mmdet.apis.inference import PartitionTwoStageDetector
        PartitionTwoStageDetector.__init__ = DummyPTSDetector.__init__
        PartitionTwoStageDetector.partition0_postprocess = \
            DummyPTSDetector.partition0_postprocess
        PartitionTwoStageDetector.partition1_postprocess = \
            DummyPTSDetector.partition1_postprocess
        PartitionTwoStageDetector.outputs0 = [torch.rand(2, 3)] * 2
        PartitionTwoStageDetector.outputs1 = [
            torch.rand(1, 9, 5), torch.rand(1, 9)
        ]
        PartitionTwoStageDetector.device_id = -1
        PartitionTwoStageDetector.CLASSES = ['' for i in range(80)]

        from mmdeploy.mmdet.apis.inference import ONNXRuntimePTSDetector
        ort_pts_detector = ONNXRuntimePTSDetector(['', ''],
                                                  ['' for i in range(80)],
                                                  model_cfg=model_cfg,
                                                  deploy_cfg=deploy_cfg,
                                                  device_id=0)

        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]
        results = ort_pts_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'ONNXRuntimePTSDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNPTSDetector():
    model_cfg, deploy_cfg = prepare_model_deploy_cfgs()
    num_outs = dict(model=dict(neck=dict(num_outs=0)))
    model_cfg.update(num_outs)

    # force add backend wrapper regardless of plugins
    # make sure NCNNPTSDetector can use TRTWrapper inside itself
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    # simplify backend inference
    outputs = {
        'scores': torch.rand(1, 12, 80),
        'boxes': torch.rand(1, 12, 4),
        'cls_score': torch.rand(1, 12, 80),
        'bbox_pred': torch.rand(1, 12, 4)
    }
    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(
            outputs=outputs, model_cfg=model_cfg, deploy_cfg=deploy_cfg)

        # replace original function in PartitionTwoStageDetector
        from mmdeploy.mmdet.apis.inference import PartitionTwoStageDetector
        PartitionTwoStageDetector.__init__ = DummyPTSDetector.__init__
        PartitionTwoStageDetector.partition0_postprocess = \
            DummyPTSDetector.partition0_postprocess
        PartitionTwoStageDetector.partition1_postprocess = \
            DummyPTSDetector.partition1_postprocess
        PartitionTwoStageDetector.outputs0 = [torch.rand(2, 3)] * 2
        PartitionTwoStageDetector.outputs1 = [
            torch.rand(1, 9, 5), torch.rand(1, 9)
        ]
        PartitionTwoStageDetector.device_id = -1
        PartitionTwoStageDetector.CLASSES = ['' for i in range(80)]

        from mmdeploy.mmdet.apis.inference import NCNNPTSDetector
        ncnn_pts_detector = NCNNPTSDetector(
            [''] * 4, [''] * 80,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            device_id=0)

        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [32, 32, 3],
            'img_shape': [32, 32, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]
        results = ncnn_pts_detector.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'NCNNPTSDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_build_detector():
    _, post_processing = get_test_cfg_and_post_processing()
    model_cfg = mmcv.Config(dict(data=dict(test={'type': 'CocoDataset'})))
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(
                type='mmdet', post_processing=post_processing)))

    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.apis.utils import init_backend_model
        detector = init_backend_model([''], model_cfg, deploy_cfg, -1)
        assert detector is not None

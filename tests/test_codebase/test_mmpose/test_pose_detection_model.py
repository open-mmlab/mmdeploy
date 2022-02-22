# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_H = 192
IMAGE_W = 256

try:
    import_codebase(Codebase.MMPOSE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPOSE} is not installed.', allow_module_level=True)


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'outputs': torch.rand(1, 1, IMAGE_H, IMAGE_W),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmcv.Config(
            {'onnx_config': {
                'output_names': ['outputs']
            }})

        from mmdeploy.utils import load_config
        model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
        model_cfg = load_config(model_cfg_path)[0]
        from mmdeploy.codebase.mmpose.deploy.pose_detection_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''],
            device='cpu',
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        img = torch.rand(1, 3, IMAGE_H, IMAGE_W)
        img_metas = [{
            'image_file':
            'tests/test_codebase/test_mmpose' + '/data/imgs/dataset/blank.jpg',
            'center': torch.tensor([0.5, 0.5]),
            'scale': 1.,
            'location': torch.tensor([0.5, 0.5]),
            'bbox_score': 0.5
        }]
        results = self.end2end_model.forward(img, img_metas)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_H, IMAGE_W)
        results = self.end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)

    def test_show_result(self):
        input_img = np.zeros([IMAGE_H, IMAGE_W, 3])
        img_path = NamedTemporaryFile(suffix='.jpg').name

        pred_bbox = torch.rand(1, 5)
        pred_keypoint = torch.rand((1, 10, 2))
        result = [{'bbox': pred_bbox, 'keypoints': pred_keypoint}]
        self.end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_pose_detection_model():
    from mmdeploy.utils import load_config
    model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type=Codebase.MMPOSE.value)))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmpose.deploy.pose_detection_model import (
            End2EndModel, build_pose_detection_model)
        posedetector = build_pose_detection_model([''], model_cfg, deploy_cfg,
                                                  'cpu')
        assert isinstance(posedetector, End2EndModel)

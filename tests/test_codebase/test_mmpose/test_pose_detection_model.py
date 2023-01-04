# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_H = 192
IMAGE_W = 256


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'outputs': torch.rand(1, 1, IMAGE_H, IMAGE_W),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config(
                {'onnx_config': {
                    'output_names': ['outputs']
                }})

            from mmdeploy.utils import load_config
            model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
            model_cfg = load_config(model_cfg_path)[0]
            from mmdeploy.codebase.mmpose.deploy.pose_detection_model import \
                End2EndModel
            model = End2EndModel(
                Backend.ONNXRUNTIME, [''],
                device='cpu',
                deploy_cfg=deploy_cfg,
                model_cfg=model_cfg)
            yield model

    def test_forward(self, end2end_model):
        img = torch.rand(1, 3, IMAGE_H, IMAGE_W)
        img_metas = [{
            'image_file':
            'tests/test_codebase/test_mmpose' + '/data/imgs/dataset/blank.jpg',
            'center': torch.tensor([0.5, 0.5]),
            'scale': 1.,
            'location': torch.tensor([0.5, 0.5]),
            'bbox_score': 0.5
        }]
        results = end2end_model.forward(img, img_metas)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_H, IMAGE_W)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)

    def test_show_result(self, end2end_model, tmp_path):
        input_img = np.zeros([IMAGE_H, IMAGE_W, 3])
        img_path = str(tmp_path / 'tmp.jpg')

        pred_bbox = torch.rand(1, 5)
        pred_keypoint = torch.rand((1, 10, 2))
        result = [{'bbox': pred_bbox, 'keypoints': pred_keypoint}]
        end2end_model.show_result(
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

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmpose.deploy.pose_detection_model import (
            End2EndModel, build_pose_detection_model)
        posedetector = build_pose_detection_model([''], model_cfg, deploy_cfg,
                                                  'cpu')
        assert isinstance(posedetector, End2EndModel)

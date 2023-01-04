# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_SIZE = 32


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'outputs': torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config(
                {'onnx_config': {
                    'output_names': ['outputs']
                }})
            model_cfg_path = 'tests/test_codebase/test_mmocr/data/dbnet.py'
            model_cfg = load_config(model_cfg_path)[0]

            from mmdeploy.codebase.mmocr.deploy.text_detection_model import \
                End2EndModel
            model = End2EndModel(
                Backend.ONNXRUNTIME, [''],
                device='cpu',
                deploy_cfg=deploy_cfg,
                model_cfg=model_cfg)
            yield model

    @pytest.mark.parametrize(
        'ori_shape',
        [[IMAGE_SIZE, IMAGE_SIZE, 3], [2 * IMAGE_SIZE, 2 * IMAGE_SIZE, 3]])
    def test_forward(self, ori_shape, end2end_model):
        imgs = [torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)]
        img_metas = [[{
            'ori_shape': ori_shape,
            'img_shape': [IMAGE_SIZE, IMAGE_SIZE, 3],
            'scale_factor': [1., 1., 1., 1.],
            'filename': ''
        }]]
        results = end2end_model.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], torch.Tensor)

    def test_show_result(self, end2end_model, tmp_path):
        input_img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
        img_path = str(tmp_path / 'tmp.jpg')

        result = {'boundary_result': [[1, 2, 3, 4, 5], [2, 2, 0, 4, 5]]}
        end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_text_detection_model():
    model_cfg_path = 'tests/test_codebase/test_mmocr/data/dbnet.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmocr')))

    from mmdeploy.backend.onnxruntime import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmocr.deploy.text_detection_model import (
            End2EndModel, build_text_detection_model)
        segmentor = build_text_detection_model([''], model_cfg, deploy_cfg,
                                               'cpu')
        assert isinstance(segmentor, End2EndModel)

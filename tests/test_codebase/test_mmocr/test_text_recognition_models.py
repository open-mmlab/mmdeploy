# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMOCR)
except ImportError:
    pytest.skip(f'{Codebase.MMOCR} is not installed.', allow_module_level=True)

IMAGE_SIZE = 32


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
            'output': torch.rand(1, 9, 37),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmcv.Config({'onnx_config': {'output_names': ['output']}})
        model_cfg_path = 'tests/test_codebase/test_mmocr/data/crnn.py'
        model_cfg = load_config(model_cfg_path)[0]

        from mmdeploy.codebase.mmocr.deploy.text_recognition_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''],
            device='cpu',
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    @pytest.mark.parametrize(
        'ori_shape',
        [[IMAGE_SIZE, IMAGE_SIZE, 3], [2 * IMAGE_SIZE, 2 * IMAGE_SIZE, 3]])
    def test_forward(self, ori_shape):
        imgs = [torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)]
        img_metas = [[{
            'ori_shape': ori_shape,
            'img_shape': [IMAGE_SIZE, IMAGE_SIZE, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]
        results = self.end2end_model.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        img_metas = [{}]
        results = self.end2end_model.forward_test(imgs, img_metas)
        assert isinstance(results[0], dict)

    def test_show_result(self):
        input_img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
        img_path = NamedTemporaryFile(suffix='.jpg').name

        result = {'text': 'sier', 'score': [0.29, 0.62, 0.25, 0.54]}
        self.end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_text_recognition_model():
    model_cfg_path = 'tests/test_codebase/test_mmocr/data/crnn.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmocr')))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmocr.deploy.text_recognition_model import (
            End2EndModel, build_text_recognition_model)
        segmentor = build_text_recognition_model([''], model_cfg, deploy_cfg,
                                                 'cpu')
        assert isinstance(segmentor, End2EndModel)

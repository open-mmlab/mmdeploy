# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine import Config
from mmengine.structures import BaseDataElement

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMROTATE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMROTATE} is not installed.', allow_module_level=True)

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
            'dets': torch.rand(1, 10, 6),
            'labels': torch.rand(1, 10)
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config(
            {'onnx_config': {
                'output_names': ['dets', 'labels']
            }})

        from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''], device='cpu', deploy_cfg=deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        imgs = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        img_metas = [
            BaseDataElement(metainfo={
                'img_shape': [IMAGE_SIZE, IMAGE_SIZE],
                'scale_factor': [1, 1]
            })
        ]
        results = self.end2end_model.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using End2EndModel'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_rotated_detection_model():
    model_cfg_path = 'tests/test_codebase/test_mmrotate/data/model.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            ir_config=dict(type='onnx', output_names=['dets', 'labels']),
            codebase_config=dict(type='mmrotate')))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import (
            End2EndModel, build_rotated_detection_model)
        segmentor = build_rotated_detection_model([''], deploy_cfg, 'cpu')
        assert isinstance(segmentor, End2EndModel)

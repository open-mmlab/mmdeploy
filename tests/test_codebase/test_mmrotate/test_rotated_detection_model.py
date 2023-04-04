# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine import Config
from mmengine.structures import BaseDataElement

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

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime.wrapper import ORTWrapper
        from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import \
            End2EndModel

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'dets': torch.rand(1, 10, 6),
                'labels': torch.rand(1, 10)
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = Config(
                {'onnx_config': {
                    'output_names': ['dets', 'labels']
                }})

            yield End2EndModel(
                Backend.ONNXRUNTIME, [''], device='cpu', deploy_cfg=deploy_cfg)

    def test_forward(self, end2end_model):
        imgs = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        img_metas = [
            BaseDataElement(metainfo={
                'img_shape': [IMAGE_SIZE, IMAGE_SIZE],
                'scale_factor': [1, 1]
            })
        ]
        results = end2end_model.forward(imgs, img_metas)
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

    from mmdeploy.backend.onnxruntime.wrapper import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import (
            End2EndModel, build_rotated_detection_model)
        segmentor = build_rotated_detection_model([''], deploy_cfg, 'cpu')
        assert isinstance(segmentor, End2EndModel)

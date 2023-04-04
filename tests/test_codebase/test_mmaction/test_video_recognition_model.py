# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_SIZE = 224

try:
    import_codebase(Codebase.MMACTION)
except ImportError:
    pytest.skip(
        f'{Codebase.MMACTION} is not installed.', allow_module_level=True)


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime.wrapper import ORTWrapper
        from mmdeploy.codebase.mmaction.deploy.video_recognition_model import \
            End2EndModel

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'outputs': torch.rand(1, 400),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = Config({'onnx_config': {'output_names': ['outputs']}})
            model_cfg_path = 'tests/test_codebase/test_mmaction/data/model.py'
            model_cfg = load_config(model_cfg_path)[0]

            yield End2EndModel(
                Backend.ONNXRUNTIME, [''],
                device='cpu',
                deploy_cfg=deploy_cfg,
                model_cfg=model_cfg)

    def test_forward(self, end2end_model):
        inputs = torch.rand(1, 3, 3, IMAGE_SIZE, IMAGE_SIZE)
        from mmaction.structures import ActionDataSample
        data_sample = ActionDataSample(
            metainfo=dict(img_shape=(IMAGE_SIZE, IMAGE_SIZE)))
        results = end2end_model.forward(inputs, [data_sample], mode='predict')
        assert results is not None, 'failed to get output using '\
            'End2EndModel'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_video_recognition_model():
    model_cfg_path = 'tests/test_codebase/test_mmaction/data/model.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmaction')))

    from mmdeploy.backend.onnxruntime.wrapper import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmaction.deploy.video_recognition_model import (
            End2EndModel, build_video_recognition_model)
        classifier = build_video_recognition_model([''], model_cfg, deploy_cfg,
                                                   'cpu')
        assert isinstance(classifier, End2EndModel)

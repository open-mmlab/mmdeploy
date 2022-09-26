# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_H = 192
IMAGE_W = 256

try:
    import_codebase(Codebase.MMPOSE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPOSE} is not installed.', allow_module_level=True)

from .utils import generate_datasample  # noqa: E402
from .utils import generate_mmpose_deploy_config  # noqa: E402


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
            'output': torch.rand(1, 1, IMAGE_H, IMAGE_W),
        }
        cls.wrapper.set(outputs=cls.outputs)

        from mmdeploy.codebase.mmpose.deploy.pose_detection_model import \
            End2EndModel
        model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
        model_cfg = load_config(model_cfg_path)[0]
        deploy_cfg = generate_mmpose_deploy_config()
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
        data_samples = [generate_datasample((IMAGE_H, IMAGE_W))]
        results = self.end2end_model.forward(img, data_samples)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_pose_detection_model():
    model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = generate_mmpose_deploy_config()

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

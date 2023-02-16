# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeEditor can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        from mmdeploy.codebase.mmedit.deploy.super_resolution_model import \
            End2EndModel

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            wrapper.set(outputs=dict(fake_img=torch.rand(3, 32, 32)))
            deploy_cfg = mmcv.Config(
                dict(
                    onnx_config=dict(
                        input_names=['masked_img', 'mask'],
                        output_names=['fake_img'])))
            model_cfg = load_config(
                'tests/test_codebase/test_mmedit/data/inpainting_model.py')[0]
            model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu', model_cfg,
                                 deploy_cfg)
            yield model

    def test_forward(self, end2end_model):
        masked_img = np.random.rand(3, 32, 32)
        mask = np.random.randint(0, 2, (32, 32))

        results = end2end_model.forward(masked_img, mask, test_mode=False)
        assert results is not None

        results = end2end_model.forward(
            masked_img, mask, test_mode=True, gt=torch.tensor(results[0]))
        assert results is not None

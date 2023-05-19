# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMAGIC)
except ImportError:
    pytest.skip(
        f'{Codebase.MMAGIC} is not installed.', allow_module_level=True)


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeEditor can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        from mmdeploy.codebase.mmagic.deploy.super_resolution_model import \
            End2EndModel

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'outputs': torch.rand(3, 64, 64),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = Config({'onnx_config': {'output_names': ['outputs']}})
            model_cfg = 'tests/test_codebase/test_mmagic/data/model.py'
            model_cfg = load_config(model_cfg)[0]
            model = End2EndModel(
                Backend.ONNXRUNTIME, [''],
                'cpu',
                model_cfg,
                deploy_cfg,
                data_preprocessor=model_cfg.model.data_preprocessor)
            yield model

    def test_forward(self, end2end_model):
        input_img = torch.rand(1, 3, 32, 32)
        from mmagic.structures import DataSample
        img_metas = DataSample(metainfo={'ori_img_shape': [(32, 32, 3)]})
        results = end2end_model.forward(input_img, img_metas)
        assert results is not None

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine import Config
from mmengine.structures import BaseDataElement

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.utils import Backend, load_config
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        # make sure ONNXRuntimeEditor can use ORTWrapper inside itself
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'outputs': torch.rand(3, 64, 64),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({'onnx_config': {'output_names': ['outputs']}})
        model_cfg = 'tests/test_codebase/test_mmedit/data/model.py'
        model_cfg = load_config(model_cfg)[0]
        from mmdeploy.codebase.mmedit.deploy.super_resolution_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(Backend.ONNXRUNTIME, [''], 'cpu',
                                         model_cfg, deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        input_img = torch.rand(1, 3, 32, 32)
        img_metas = [BaseDataElement(metainfo={'ori_img_shape': [3, 32, 32]})]
        results = self.end2end_model.forward(input_img, img_metas)
        assert results is not None

# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest
import torch
from mmengine import Config

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

IMAGE_SIZE = 64
NUM_CLASS = 1000

try:
    import_codebase(Codebase.MMCLS)
except ImportError:
    pytest.skip(f'{Codebase.MMCLS} is not installed.', allow_module_level=True)


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
            'outputs': torch.rand(1, 100),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({'onnx_config': {'output_names': ['outputs']}})

        from mmdeploy.codebase.mmcls.deploy.classification_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''], device='cpu', deploy_cfg=deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        imgs = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        from mmcls.structures import ClsDataSample
        data_sample = ClsDataSample(
            metainfo=dict(
                scale_factor=(1, 1),
                ori_shape=(IMAGE_SIZE, IMAGE_SIZE),
                img_shape=(IMAGE_SIZE, IMAGE_SIZE)))
        results = self.end2end_model.forward(
            imgs, [data_sample], mode='predict')
        assert results is not None, 'failed to get output using '\
            'End2EndModel'


@backend_checker(Backend.RKNN)
class TestRKNNEnd2EndModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        import mmdeploy.backend.rknn as rknn_apis
        from mmdeploy.backend.rknn import RKNNWrapper
        rknn_apis.__dict__.update({'RKNNWrapper': RKNNWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(RKNNWrapper)
        cls.outputs = [torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)]
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = Config({
            'onnx_config': {
                'output_names': ['outputs']
            },
            'backend_config': {
                'common_config': {}
            }
        })

        from mmdeploy.codebase.mmcls.deploy.classification_model import \
            RKNNEnd2EndModel
        class_names = ['' for i in range(NUM_CLASS)]
        cls.end2end_model = RKNNEnd2EndModel(
            Backend.RKNN, [''],
            device='cpu',
            class_names=class_names,
            deploy_cfg=deploy_cfg)

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = self.end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)


@backend_checker(Backend.ONNXRUNTIME)
def test_build_classification_model():
    model_cfg = Config(dict(data=dict(test={'type': 'ImageNet'})))
    deploy_cfg = Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmcls')))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmcls.deploy.classification_model import (
            End2EndModel, build_classification_model)
        classifier = build_classification_model([''], model_cfg, deploy_cfg,
                                                'cpu')
        assert isinstance(classifier, End2EndModel)

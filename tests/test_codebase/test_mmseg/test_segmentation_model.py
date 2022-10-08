# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMSEG)
except ImportError:
    pytest.skip(f'{Codebase.MMSEG} is not installed.', allow_module_level=True)

from .utils import generate_datasample  # noqa: E402
from .utils import generate_mmseg_deploy_config  # noqa: E402

NUM_CLASS = 19
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
            'output': torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = generate_mmseg_deploy_config()

        from mmdeploy.codebase.mmseg.deploy.segmentation_model import \
            End2EndModel
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''], device='cpu', deploy_cfg=deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        from mmseg.structures import SegDataSample
        imgs = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        data_samples = [generate_datasample(IMAGE_SIZE, IMAGE_SIZE)]
        results = self.end2end_model.forward(imgs, data_samples)
        assert len(results) == 1
        assert isinstance(results[0], SegDataSample)


@backend_checker(Backend.RKNN)
class TestRKNNModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        import mmdeploy.backend.rknn as rknn_apis
        from mmdeploy.backend.rknn import RKNNWrapper
        rknn_apis.__dict__.update({'RKNNWrapper': RKNNWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(RKNNWrapper)
        cls.outputs = [torch.rand(1, 19, IMAGE_SIZE, IMAGE_SIZE)]
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmengine.Config({
            'onnx_config': {
                'output_names': ['outputs']
            },
            'backend_config': {
                'common_config': {}
            }
        })

        from mmdeploy.codebase.mmseg.deploy.segmentation_model import RKNNModel
        class_names = ['' for i in range(NUM_CLASS)]
        palette = np.random.randint(0, 255, size=(NUM_CLASS, 3))
        cls.rknn_model = RKNNModel(
            Backend.RKNN, [''],
            device='cpu',
            class_names=class_names,
            palette=palette,
            deploy_cfg=deploy_cfg)

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = self.rknn_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)


@backend_checker(Backend.ONNXRUNTIME)
def test_build_segmentation_model():
    model_cfg = mmengine.Config(
        dict(data=dict(test={'type': 'CityscapesDataset'})))
    deploy_cfg = generate_mmseg_deploy_config()

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmseg.deploy.segmentation_model import (
            End2EndModel, build_segmentation_model)
        segmentor = build_segmentation_model([''], model_cfg, deploy_cfg,
                                             'cpu')
        assert isinstance(segmentor, End2EndModel)

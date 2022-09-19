# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

import_codebase(Codebase.MMSEG)

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

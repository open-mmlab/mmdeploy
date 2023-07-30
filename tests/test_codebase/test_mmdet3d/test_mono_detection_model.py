# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

try:
    import_codebase(Codebase.MMDET3D)
except ImportError:
    pytest.skip(
        f'{Codebase.MMDET3D} is not installed.', allow_module_level=True)
from mmdeploy.codebase.mmdet3d.deploy.mono_detection_model import \
    MonoDetectionModel

nuscenes_pcd_path = 'tests/test_codebase/test_mmdet3d/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl'  # noqa: E501
somke_model_cfg = 'tests/test_codebase/test_mmdet3d/data/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'  # noqa: E501


@backend_checker(Backend.ONNXRUNTIME)
class TestMonoDetectionModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'cls_score': torch.rand(1, 3, 96, 320),
            'bbox_pred': torch.rand(1, 8, 96, 320),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmengine.Config({
            'onnx_config': {
                'input_names': ['input'],
                'output_names': ['cls_score', 'bbox_pred'],
                'opset_version': 11
            },
            'backend_config': {
                'type': 'onnxruntime'
            }
        })

        from mmdeploy.utils import load_config
        model_cfg_path = somke_model_cfg
        model_cfg = load_config(model_cfg_path)[0]
        cls.end2end_model = MonoDetectionModel(
            Backend.ONNXRUNTIME, [''],
            device='cuda',
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    @pytest.mark.skipif(
        reason='Only support GPU test',
        condition=not torch.cuda.is_available())
    def test_forward_and_show_result(self):
        inputs = {
            'imgs': torch.rand((1, 3, 384, 1280)),
        }
        results = self.end2end_model.forward(inputs=inputs)
        assert results is not None


@backend_checker(Backend.ONNXRUNTIME)
def test_build_mono_detection_model():
    from mmdeploy.utils import load_config
    model_cfg_path = somke_model_cfg
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            onnx_config=dict(output_names=['cls_score', 'bbox_pred']),
            codebase_config=dict(type=Codebase.MMDET3D.value)))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmdet3d.deploy.mono_detection_model import (
            MonoDetectionModel, build_mono_detection_model)
        monodetector = build_mono_detection_model([''],
                                                  model_cfg=model_cfg,
                                                  deploy_cfg=deploy_cfg,
                                                  device='cpu')
        assert isinstance(monodetector, MonoDetectionModel)

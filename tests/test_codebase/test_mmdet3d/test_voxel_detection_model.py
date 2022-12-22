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
from mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model import \
    VoxelDetectionModel

pcd_path = 'tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin'
model_cfg = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'


@backend_checker(Backend.ONNXRUNTIME)
class TestVoxelDetectionModel:

    @classmethod
    def setup_class(cls):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

        # simplify backend inference
        cls.wrapper = SwitchBackendWrapper(ORTWrapper)
        cls.outputs = {
            'cls_score': torch.rand(1, 18, 32, 32),
            'bbox_pred': torch.rand(1, 42, 32, 32),
            'dir_cls_pred': torch.rand(1, 12, 32, 32)
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmengine.Config({
            'onnx_config': {
                'input_names': ['voxels', 'num_points', 'coors'],
                'output_names': ['cls_score', 'bbox_pred', 'dir_cls_pred'],
                'opset_version': 11
            },
            'backend_config': {
                'type': 'onnxruntime'
            }
        })

        from mmdeploy.utils import load_config
        model_cfg_path = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'
        model_cfg = load_config(model_cfg_path)[0]
        cls.end2end_model = VoxelDetectionModel(
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
            'voxels': {
                'voxels': torch.rand((3945, 32, 4)),
                'num_points': torch.ones((3945), dtype=torch.int32),
                'coors': torch.ones((3945, 4), dtype=torch.int32)
            }
        }
        results = self.end2end_model.forward(inputs=inputs)
        assert results is not None


@backend_checker(Backend.ONNXRUNTIME)
def test_build_voxel_detection_model():
    from mmdeploy.utils import load_config
    model_cfg_path = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            onnx_config=dict(
                output_names=['cls_score', 'bbox_pred', 'dir_cls_pred']),
            codebase_config=dict(type=Codebase.MMDET3D.value)))

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model import (
            VoxelDetectionModel, build_voxel_detection_model)
        voxeldetector = build_voxel_detection_model([''], model_cfg,
                                                    deploy_cfg, 'cpu')
        assert isinstance(voxeldetector, VoxelDetectionModel)

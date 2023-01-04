# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import pytest
import torch

from mmdeploy.codebase.mmdet3d.deploy.voxel_detection import VoxelDetection
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

pcd_path = 'tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin'
model_cfg = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'


@backend_checker(Backend.ONNXRUNTIME)
class TestVoxelDetectionModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper
        from mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model import \
            VoxelDetectionModel
        from mmdeploy.utils import load_config

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'bboxes': torch.rand(1, 50, 7),
                'scores': torch.rand(1, 50),
                'labels': torch.rand(1, 50)
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
                'onnx_config': {
                    'input_names': ['voxels', 'num_points', 'coors'],
                    'output_names': ['bboxes', 'scores', 'labels'],
                    'opset_version': 11
                },
                'backend_config': {
                    'type': 'tensorrt'
                }
            })

            model_cfg_path = 'tests/test_codebase/test_mmdet3d/data' + \
                '/model_cfg.py'
            model_cfg = load_config(model_cfg_path)[0]

            model = VoxelDetectionModel(
                Backend.ONNXRUNTIME, [''],
                device='cuda',
                deploy_cfg=deploy_cfg,
                model_cfg=model_cfg)
            yield model

    @pytest.mark.skipif(
        reason='Only support GPU test',
        condition=not torch.cuda.is_available())
    def test_forward_and_show_result(self, end2end_model, tmp_path):
        data = VoxelDetection.read_pcd_file(pcd_path, model_cfg, 'cuda')
        results = end2end_model.forward(data['points'], data['img_metas'])
        assert results is not None
        dir = str(tmp_path)
        end2end_model.show_result(
            data, results, dir, 'backend_output.bin', show=False)
        assert osp.exists(dir + '/backend_output.bin')


@backend_checker(Backend.ONNXRUNTIME)
def test_build_voxel_detection_model():
    from mmdeploy.utils import load_config
    model_cfg_path = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'
    model_cfg = load_config(model_cfg_path)[0]
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            onnx_config=dict(
                output_names=['scores', 'bbox_preds', 'dir_scores']),
            codebase_config=dict(type=Codebase.MMDET3D.value)))

    from mmdeploy.backend.onnxruntime import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model import (
            VoxelDetectionModel, build_voxel_detection_model)
        voxeldetector = build_voxel_detection_model([''], model_cfg,
                                                    deploy_cfg, 'cpu')
        assert isinstance(voxeldetector, VoxelDetectionModel)

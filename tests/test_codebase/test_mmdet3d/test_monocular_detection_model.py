# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import pytest
import torch

from mmdeploy.codebase.mmdet3d.deploy.monocular_detection_model import (
    MonocularDetectionModel, build_monocular_detection_model)
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker


@backend_checker(Backend.ONNXRUNTIME)
class TestMonocularDetectionModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        num_classes = 10
        num_attr = 5
        num_dets = 20
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'bboxes': torch.rand(1, num_dets, 9),
                'scores': torch.rand(1, num_dets),
                'labels': torch.randint(num_classes, (1, num_dets)),
                'dir_scores': torch.randint(2, (1, num_dets)),
                'attrs': torch.randint(num_attr, (1, num_dets))
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
                'onnx_config': {
                    'input_names': ['img', 'cam2img', 'cam2img_inverse'],
                    'output_names':
                    ['bboxes', 'scores', 'labels', 'dir_scores', 'attrs'],
                    'opset_version':
                    11
                },
                'backend_config': {
                    'type': 'tensorrt'
                }
            })

            model = MonocularDetectionModel(
                Backend.ONNXRUNTIME,
                [''],
                device='cuda',
                model_cfg=['' for i in range(10)],
                deploy_cfg=deploy_cfg,
            )
            yield model

    @pytest.mark.skipif(
        reason='Only support GPU test',
        condition=not torch.cuda.is_available())
    def test_forward_and_show_result(self, end2end_model, tmp_path):
        from mmdet3d.core import Box3DMode
        from mmdet3d.core.bbox.structures.box_3d_mode import \
            CameraInstance3DBoxes
        img = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'filename':
            'tests/test_codebase/test_mmdet3d/data/nuscenes/'
            'n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg',
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'cam2img': [[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
            'box_type_3d':
            CameraInstance3DBoxes,
            'box_mode_3d':
            Box3DMode.CAM,
        }]]
        data = dict(img=img, img_metas=img_metas)
        results = end2end_model.forward(img, img_metas)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == 1
        # assert results[0]['img_bbox']['scores_3d'].shape == 4
        dir = str(tmp_path)
        end2end_model.show_result(data, results,
                                  osp.join(dir, 'backend_output'))
        assert osp.exists(dir + '/backend_output')


@backend_checker(Backend.ONNXRUNTIME)
def test_build_monocular_detection_model():
    model_cfg = mmcv.Config(
        dict(data=dict(test={'type': 'NuScenesMonoDataset'})))
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            onnx_config=dict(output_names=[
                'bboxes', 'scores', 'labels', 'dir_scores', 'attrs'
            ]),
            codebase_config=dict(type=Codebase.MMDET3D.value)))

    from mmdeploy.backend.onnxruntime import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        monoculardetector = build_monocular_detection_model([''], model_cfg,
                                                            deploy_cfg, 'cpu')
        assert isinstance(monoculardetector, MonocularDetectionModel)

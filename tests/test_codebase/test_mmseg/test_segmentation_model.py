# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

NUM_CLASS = 19
IMAGE_SIZE = 32


@backend_checker(Backend.ONNXRUNTIME)
class TestEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.onnxruntime import ORTWrapper

        # simplify backend inference
        with SwitchBackendWrapper(ORTWrapper) as wrapper:
            outputs = {
                'outputs': torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE),
            }
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config(
                {'onnx_config': {
                    'output_names': ['outputs']
                }})

            from mmdeploy.codebase.mmseg.deploy.segmentation_model import \
                End2EndModel
            class_names = ['' for i in range(NUM_CLASS)]
            palette = np.random.randint(0, 255, size=(NUM_CLASS, 3))
            model = End2EndModel(
                Backend.ONNXRUNTIME, [''],
                device='cpu',
                class_names=class_names,
                palette=palette,
                deploy_cfg=deploy_cfg)
            yield model

    @pytest.mark.parametrize(
        'ori_shape',
        [[IMAGE_SIZE, IMAGE_SIZE, 3], [2 * IMAGE_SIZE, 2 * IMAGE_SIZE, 3]])
    def test_forward(self, ori_shape, end2end_model):
        imgs = [torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)]
        img_metas = [[{
            'ori_shape': ori_shape,
            'img_shape': [IMAGE_SIZE, IMAGE_SIZE, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]
        results = end2end_model.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)

    def test_show_result(self, end2end_model, tmp_path):
        input_img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
        img_path = str(tmp_path / 'tmp.jpg')

        result = [torch.rand(IMAGE_SIZE, IMAGE_SIZE)]
        end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


@backend_checker(Backend.RKNN)
class TestRKNNModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.rknn import RKNNWrapper
        from mmdeploy.codebase.mmseg.deploy.segmentation_model import RKNNModel

        # simplify backend inference
        with SwitchBackendWrapper(RKNNWrapper) as wrapper:
            outputs = [torch.rand(1, 19, IMAGE_SIZE, IMAGE_SIZE)]
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
                'onnx_config': {
                    'output_names': ['outputs']
                },
                'backend_config': {
                    'common_config': {}
                }
            })

            class_names = ['' for i in range(NUM_CLASS)]
            palette = np.random.randint(0, 255, size=(NUM_CLASS, 3))
            model = RKNNModel(
                Backend.RKNN, [''],
                device='cpu',
                class_names=class_names,
                palette=palette,
                deploy_cfg=deploy_cfg)
            yield model

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)


@pytest.mark.parametrize('from_file', [True, False])
@pytest.mark.parametrize('data_type', ['train', 'val', 'test'])
def test_get_classes_palette_from_config(from_file, data_type, tmp_path):
    from mmseg.datasets import DATASETS

    from mmdeploy.codebase.mmseg.deploy.segmentation_model import \
        get_classes_palette_from_config
    dataset_type = 'CityscapesDataset'
    data_cfg = mmcv.Config({
        'data': {
            data_type:
            dict(
                type=dataset_type,
                data_root='',
                img_dir='',
                ann_dir='',
                pipeline=None)
        }
    })

    if from_file:
        config_path = str(tmp_path / 'tmp_cfg.py')
        with open(config_path, 'w') as file:
            file.write(data_cfg.pretty_text)
        data_cfg = config_path

    classes, palette = get_classes_palette_from_config(data_cfg)
    module = DATASETS.module_dict[dataset_type]
    assert classes == module.CLASSES, \
        f'fail to get CLASSES of dataset: {dataset_type}'
    assert palette == module.PALETTE, \
        f'fail to get PALETTE of dataset: {dataset_type}'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_segmentation_model():
    model_cfg = mmcv.Config(
        dict(data=dict(test={'type': 'CityscapesDataset'})))
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmseg')))

    from mmdeploy.backend.onnxruntime import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmseg.deploy.segmentation_model import (
            End2EndModel, build_segmentation_model)
        segmentor = build_segmentation_model([''], model_cfg, deploy_cfg,
                                             'cpu')
        assert isinstance(segmentor, End2EndModel)

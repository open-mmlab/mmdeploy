# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

NUM_CLASS = 1000
IMAGE_SIZE = 64


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

            from mmdeploy.codebase.mmcls.deploy.classification_model import \
                End2EndModel
            class_names = ['' for i in range(NUM_CLASS)]
            model = End2EndModel(
                Backend.ONNXRUNTIME, [''],
                device='cpu',
                class_names=class_names,
                deploy_cfg=deploy_cfg)
            yield model

    def test_forward(self, end2end_model):
        imgs = [torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)]
        results = end2end_model.forward(imgs)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)

    def test_show_result(self, end2end_model, tmp_path):
        input_img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
        img_path = str(tmp_path / 'tmp.jpg')

        pred_label = torch.randint(0, NUM_CLASS, (1, ))
        pred_score = torch.rand((1, ))
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


@backend_checker(Backend.RKNN)
class TestRKNNEnd2EndModel:

    @pytest.fixture(scope='class')
    def end2end_model(self):
        # force add backend wrapper regardless of plugins
        from mmdeploy.backend.rknn import RKNNWrapper

        # simplify backend inference
        with SwitchBackendWrapper(RKNNWrapper) as wrapper:
            outputs = [torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)]
            wrapper.set(outputs=outputs)
            deploy_cfg = mmcv.Config({
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
            model = RKNNEnd2EndModel(
                Backend.RKNN, [''],
                device='cpu',
                class_names=class_names,
                deploy_cfg=deploy_cfg)
            yield model

    def test_forward_test(self, end2end_model):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)


@pytest.mark.parametrize('from_file', [True, False])
@pytest.mark.parametrize('data_type', ['train', 'val', 'test'])
def test_get_classes_from_config(from_file, data_type, tmp_path):
    from mmcls.datasets import DATASETS

    from mmdeploy.codebase.mmcls.deploy.classification_model import \
        get_classes_from_config
    dataset_type = 'ImageNet'
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
        config_path = str(tmp_path / 'tmp.py')
        with open(config_path, 'w') as file:
            file.write(data_cfg.pretty_text)
        data_cfg = config_path

    classes = get_classes_from_config(data_cfg)
    module = DATASETS.module_dict[dataset_type]
    assert classes == module.CLASSES, \
        f'fail to get CLASSES of dataset: {dataset_type}'


@backend_checker(Backend.ONNXRUNTIME)
def test_build_classification_model():
    model_cfg = mmcv.Config(dict(data=dict(test={'type': 'ImageNet'})))
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            onnx_config=dict(output_names=['outputs']),
            codebase_config=dict(type='mmcls')))

    from mmdeploy.backend.onnxruntime import ORTWrapper

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmcls.deploy.classification_model import (
            End2EndModel, build_classification_model)
        classifier = build_classification_model([''], model_cfg, deploy_cfg,
                                                'cpu')
        assert isinstance(classifier, End2EndModel)

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper, backend_checker

NUM_CLASS = 1000
IMAGE_SIZE = 64

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
            'outputs': torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        }
        cls.wrapper.set(outputs=cls.outputs)
        deploy_cfg = mmcv.Config(
            {'onnx_config': {
                'output_names': ['outputs']
            }})

        from mmdeploy.codebase.mmcls.deploy.classification_model import \
            End2EndModel
        class_names = ['' for i in range(NUM_CLASS)]
        cls.end2end_model = End2EndModel(
            Backend.ONNXRUNTIME, [''],
            device='cpu',
            class_names=class_names,
            deploy_cfg=deploy_cfg)

    @classmethod
    def teardown_class(cls):
        cls.wrapper.recover()

    def test_forward(self):
        imgs = [torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)]
        results = self.end2end_model.forward(imgs)
        assert results is not None, 'failed to get output using '\
            'End2EndModel'

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = self.end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)

    def test_show_result(self):
        input_img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3])
        img_path = NamedTemporaryFile(suffix='.jpg').name

        pred_label = torch.randint(0, NUM_CLASS, (1, ))
        pred_score = torch.rand((1, ))
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        self.end2end_model.show_result(
            input_img, result, '', show=False, out_file=img_path)
        assert osp.exists(img_path), 'Fails to create drawn image.'


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
        cls.end2end_model = RKNNEnd2EndModel(
            Backend.RKNN, [''],
            device='cpu',
            class_names=class_names,
            deploy_cfg=deploy_cfg)

    def test_forward_test(self):
        imgs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        results = self.end2end_model.forward_test(imgs)
        assert isinstance(results[0], np.ndarray)


@pytest.mark.parametrize('from_file', [True, False])
@pytest.mark.parametrize('data_type', ['train', 'val', 'test'])
def test_get_classes_from_config(from_file, data_type):
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
        config_path = NamedTemporaryFile(suffix='.py').name
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
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.codebase.mmcls.deploy.classification_model import (
            End2EndModel, build_classification_model)
        classifier = build_classification_model([''], model_cfg, deploy_cfg,
                                                'cpu')
        assert isinstance(classifier, End2EndModel)

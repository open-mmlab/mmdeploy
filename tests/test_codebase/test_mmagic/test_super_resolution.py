# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch
from mmengine import Config
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import mmdeploy.apis.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.core.rewriters.rewriter_manager import RewriterContext
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper, WrapFunction

try:
    import_codebase(Codebase.MMAGIC)
except ImportError:
    pytest.skip(
        f'{Codebase.MMAGIC} is not installed.', allow_module_level=True)

model_cfg = 'tests/test_codebase/test_mmagic/data/model.py'
model_cfg = load_config(model_cfg)[0]
deploy_cfg = Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmagic', task='SuperResolution'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))
input_img = np.random.rand(32, 32, 3)
img_shape = [32, 32]
input = {'img': input_img}
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = None


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, 3, 50, 50),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_test_runner():
    # Prepare dummy model
    from mmagic.structures import DataSample

    img_meta = dict(ori_img_shape=(32, 32, 3))
    img = torch.rand(3, 32, 32)
    data_sample = DataSample(gt_img=img, metainfo=img_meta)
    data_sample.set_data(
        dict(output=DataSample(pred_img=img, metainfo=img_meta)))
    data_sample.set_data(dict(input=img))
    outputs = [data_sample]
    model = DummyModel(outputs=outputs)
    assert model is not None
    # Run test
    with TemporaryDirectory() as dir:
        runner = task_processor.build_test_runner(model, dir)
        wrapped_func = WrapFunction(runner.test)

        with RewriterContext({}):
            _ = wrapped_func()


def test_build_pytorch_model():
    from mmagic.models import BaseEditModel
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseEditModel)


def test_build_backend_model(backend_model):
    assert backend_model is not None


def test_create_input():
    inputs = task_processor.create_input(input_img, input_shape=img_shape)
    assert inputs is not None


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(input_img, img_shape)

    with torch.no_grad():
        results = backend_model.test_step(input_dict)[0]

    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(input_img, results, filename, 'window')
        assert os.path.exists(filename)


def test_get_tensor_from_input():
    assert type(task_processor.get_tensor_from_input(input)) is not dict


def test_get_partition_cfg():
    with pytest.raises(NotImplementedError):
        task_processor.get_partition_cfg(None)


def test_build_dataset_and_dataloader():
    data = dict(
        type='BasicImageDataset',
        ann_file='test_ann.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root='tests/test_codebase/test_mmagic/data',
        data_prefix=dict(img='imgs', gt='imgs'),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                key='img',
                color_type='color',
                channel_order='rgb',
                imdecode_backend='cv2'),
        ])
    dataset = task_processor.build_dataset(dataset_cfg=data)
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader_cfg = dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=data)
    dataloader = task_processor.build_dataloader(dataloader_cfg)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'

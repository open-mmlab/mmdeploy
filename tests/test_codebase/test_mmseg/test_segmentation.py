# Copyright (c) OpenMMLab. All rights reserved.
import copy
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import mmcv
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    import_codebase(Codebase.MMSEG)
except ImportError:
    pytest.skip(f'{Codebase.MMSEG} is not installed.', allow_module_level=True)

from .utils import generate_datasample  # noqa: E402
from .utils import generate_mmseg_deploy_config  # noqa: E402

model_cfg_path = 'tests/test_codebase/test_mmseg/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = generate_mmseg_deploy_config()

task_processor = None
img_shape = (32, 32)
tiger_img_path = 'tests/data/tiger.jpeg'
img = mmcv.imread(tiger_img_path)
img = mmcv.imresize(img, img_shape)


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.mark.parametrize('from_mmrazor', [True, False, '123', 0])
def test_build_pytorch_model(from_mmrazor: Any):
    from mmseg.models.segmentors.base import BaseSegmentor
    if from_mmrazor is False:
        _task_processor = task_processor
    else:
        _model_cfg_path = 'tests/test_codebase/test_mmseg/data/' \
            'mmrazor_model.py'
        _model_cfg = load_config(_model_cfg_path)[0]
        _model_cfg.algorithm.architecture.model.type = 'mmseg.EncoderDecoder'
        _model_cfg.algorithm.distiller.teacher.type = 'mmseg.EncoderDecoder'
        _deploy_cfg = copy.deepcopy(deploy_cfg)
        _deploy_cfg.codebase_config['from_mmrazor'] = from_mmrazor
        _task_processor = build_task_processor(_model_cfg, _deploy_cfg, 'cpu')

    if not isinstance(from_mmrazor, bool):
        with pytest.raises(
                TypeError,
                match='`from_mmrazor` attribute must be '
                'boolean type! '
                f'but got: {from_mmrazor}'):
            _ = _task_processor.from_mmrazor
        return
    assert from_mmrazor == _task_processor.from_mmrazor
    if from_mmrazor:
        pytest.importorskip('mmrazor', reason='mmrazor is not installed.')
    model = _task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseSegmentor)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, 1, *img_shape),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


def test_create_input():
    img_path = 'tests/data/tiger.jpeg'
    data_preprocessor = task_processor.build_data_preprocessor()
    inputs = task_processor.create_input(
        img_path, input_shape=img_shape, data_preprocessor=data_preprocessor)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_build_data_preprocessor():
    from mmseg.models import SegDataPreProcessor
    data_preprocessor = task_processor.build_data_preprocessor()
    assert isinstance(data_preprocessor, SegDataPreProcessor)


def test_get_visualizer():
    from mmseg.visualization import SegLocalVisualizer
    tmp_dir = TemporaryDirectory().name
    visualizer = task_processor.get_visualizer('ort', tmp_dir)
    assert isinstance(visualizer, SegLocalVisualizer)


def test_get_tensort_from_input():
    data = torch.rand(3, 4, 5)
    input_data = {'inputs': data}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, data)


def test_get_partition_cfg():
    try:
        _ = task_processor.get_partition_cfg(partition_type='')
    except NotImplementedError:
        pass


def test_build_dataset_and_dataloader():
    from torch.utils.data import DataLoader, Dataset
    val_dataloader = model_cfg['val_dataloader']
    dataset = task_processor.build_dataset(
        dataset_cfg=val_dataloader['dataset'])
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(val_dataloader)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_build_test_runner(backend_model):
    from mmdeploy.codebase.base.runner import DeployTestRunner
    temp_dir = TemporaryDirectory().name
    runner = task_processor.build_test_runner(backend_model, temp_dir)
    assert isinstance(runner, DeployTestRunner)


def test_visualize():
    h, w = img.shape[:2]
    datasample = generate_datasample(h, w)
    output_file = NamedTemporaryFile(suffix='.jpg').name
    task_processor.visualize(
        img, datasample, output_file, show_result=False, window_name='test')


def test_get_preprocess():
    process = task_processor.get_preprocess()
    assert process is not None


def test_get_postprocess():
    process = task_processor.get_postprocess()
    assert isinstance(process, dict)


def test_get_model_name():
    name = task_processor.get_model_name()
    assert isinstance(name, str)

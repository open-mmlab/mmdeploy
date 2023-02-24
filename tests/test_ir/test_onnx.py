# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

import pytest

from mmdeploy.ir.onnx import ONNXIRParam, ONNXManager


@pytest.fixture(scope='class')
def file_name():
    return 'tmp'


@pytest.fixture(scope='class')
def dummy_model():
    import torch

    class DummyModel(torch.nn.Module):

        def forward(self, x, y):
            return x + y, x - y

    return DummyModel()


@pytest.fixture(scope='class')
def dummy_args():
    import torch
    return (torch.ones([2, 2]), torch.ones([2, 2]) + 1)


@pytest.fixture
def dummy_param(dummy_args, tmp_path, file_name):
    return ONNXIRParam(
        args=dummy_args,
        work_dir=str(tmp_path),
        file_name=file_name,
        input_names=['x', 'y'],
        output_names=['out0', 'out1'])


@pytest.mark.skipif(
    importlib.util.find_spec('torch') is None, reason='PyTorch is required.')
class TestONNXIRParam:

    def test_file_name(self, dummy_param, file_name):
        assert dummy_param.file_name == f'{file_name}.onnx'

    def test_check(self, dummy_param):
        dummy_param.check()

        with pytest.raises(AssertionError):
            dummy_param.opset_version = 6
            dummy_param.check()


@pytest.mark.skipif(
    importlib.util.find_spec('torch') is None, reason='PyTorch is required.')
class TestONNXManager:

    def test_build_param(self, dummy_args, file_name):
        assert isinstance(
            ONNXManager.build_params(args=dummy_args, file_name=file_name),
            ONNXIRParam)

    def test_is_available(self):
        assert ONNXManager.is_available()

    def test_export(self, dummy_model, dummy_args, file_name, tmp_path):
        output_path = str(tmp_path / f'{file_name}.onnx')
        ONNXManager.export(dummy_model, dummy_args, output_path)
        assert osp.exists(output_path)

    def test_export_from_param(self, dummy_model, dummy_param):
        output_path = osp.join(dummy_param.work_dir, dummy_param.file_name)
        ONNXManager.export_from_param(dummy_model, dummy_param)
        assert osp.exists(output_path)

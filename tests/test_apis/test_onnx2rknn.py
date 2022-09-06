# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import pytest
import torch
import torch.nn as nn

from mmdeploy.utils import Backend
from mmdeploy.utils.test import backend_checker

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand([1, 3, 8, 8])


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval()


def generate_onnx_file(model):
    with torch.no_grad():
        torch.onnx.export(
            model,
            test_img,
            onnx_file,
            output_names=['output'],
            input_names=['input'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)
        assert osp.exists(onnx_file)


def get_deploy_cfg():
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(
                type='rknn',
                common_config=dict(),
                quantization_config=dict(do_quantization=False, dataset=None),
                input_size_list=[[3, 8, 8]])))
    return deploy_cfg


@backend_checker(Backend.RKNN)
def test_onnx2rknn():
    from mmdeploy.backend.rknn.onnx2rknn import onnx2rknn
    model = test_model
    generate_onnx_file(model)

    work_dir, _ = osp.split(onnx_file)
    rknn_file = onnx_file.replace('.onnx', '.rknn')
    deploy_cfg = get_deploy_cfg()
    onnx2rknn(onnx_file, rknn_file, deploy_cfg)
    assert osp.exists(work_dir)
    assert osp.exists(rknn_file)

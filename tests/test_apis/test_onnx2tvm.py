# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

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
        self.conv = torch.nn.Conv2d(3, 8, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


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


@backend_checker(Backend.TVM)
def test_onnx2tvm():
    from mmdeploy.apis.tvm import from_onnx, get_library_ext
    model = test_model
    generate_onnx_file(model)

    work_dir, _ = osp.split(onnx_file)
    file_name = osp.splitext(onnx_file)[0]
    ext = get_library_ext()
    lib_path = osp.join(work_dir, file_name + ext)
    bytecode_path = osp.join(work_dir, file_name + '.code')
    log_file = osp.join(work_dir, file_name + '.log')
    shape = {'input': test_img.shape}
    dtype = {'input': 'float32'}
    target = 'llvm'

    # test default tuner
    tuner_dict = dict(type='DefaultTuner', target=target)
    from_onnx(onnx_file, lib_path, shape=shape, dtype=dtype, tuner=tuner_dict)
    assert osp.exists(lib_path)

    # test autotvm
    lib_path = osp.join(work_dir, file_name + '_autotvm' + ext)
    bytecode_path = osp.join(work_dir, file_name + '_autotvm.code')
    log_file = osp.join(work_dir, file_name + '_autotvm.log')
    tuner_dict = dict(
        type='AutoTVMTuner',
        target=target,
        log_file=log_file,
        n_trial=1,
        tuner=dict(type='XGBTuner'))
    from_onnx(
        onnx_file,
        lib_path,
        use_vm=True,
        bytecode_file=bytecode_path,
        shape=shape,
        dtype=dtype,
        tuner=tuner_dict)
    assert osp.exists(lib_path)
    assert osp.exists(bytecode_path)

    # test ansor
    lib_path = osp.join(work_dir, file_name + '_ansor' + ext)
    bytecode_path = osp.join(work_dir, file_name + '_ansor.code')
    log_file = osp.join(work_dir, file_name + '_ansor.log')
    tuner_dict = dict(
        type='AutoScheduleTuner',
        target=target,
        log_file=log_file,
        num_measure_trials=2)
    from_onnx(
        onnx_file,
        lib_path,
        use_vm=True,
        bytecode_file=bytecode_path,
        shape=shape,
        dtype=dtype,
        tuner=tuner_dict)
    assert osp.exists(lib_path)
    assert osp.exists(bytecode_path)

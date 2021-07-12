import os
import os.path as osp
import shutil

import mmcv
import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

import mmdeploy.apis.tensorrt as trt_apis

# skip if tensorrt apis can not loaded
if not trt_apis.is_available():
    pytest.skip('TensorRT apis is not prepared.')
trt = pytest.importorskip('tensorrt', reason='Import tensorrt failed.')
if not torch.cuda.is_available():
    pytest.skip('CUDA is not available.')

# load apis from trt_apis
TRTWrapper = trt_apis.TRTWrapper
onnx2tensorrt = trt_apis.onnx2tensorrt

ret_value = mp.Value('d', 0, lock=False)
work_dir = './tmp/'
onnx_file = 'tmp.onnx'
save_file = 'tmp.engine'


@pytest.fixture(autouse=True)
def clear_workdir_after_test():
    # clear work_dir before test
    if osp.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)

    yield

    # clear work_dir after test
    if osp.exists(work_dir):
        shutil.rmtree(work_dir)


def test_onnx2tensorrt():

    # dummy model
    class TestModel(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    model = TestModel().eval().cuda()
    x = torch.rand(1, 3, 64, 64).cuda()

    onnx_path = osp.join(work_dir, onnx_file)
    # export to onnx
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        }})

    assert osp.exists(onnx_path)

    # deploy config
    deploy_cfg = mmcv.Config(
        dict(
            backend='tensorrt',
            tensorrt_params=dict(
                shared_params=dict(
                    log_level=trt.Logger.WARNING, fp16_mode=False),
                model_params=[
                    dict(
                        opt_shape_dict=dict(
                            input=[[1, 3, 32, 32], [1, 3, 64, 64],
                                   [1, 3, 128, 128]]),
                        max_workspace_size=1 << 30)
                ])))

    # convert to engine
    onnx2tensorrt(
        work_dir,
        save_file,
        0,
        deploy_cfg=deploy_cfg,
        onnx_model=onnx_path,
        ret_value=ret_value)

    assert ret_value.value == 0
    assert osp.exists(work_dir)
    assert osp.exists(osp.join(work_dir, save_file))

    # test
    trt_model = TRTWrapper(osp.join(work_dir, save_file))
    x = x.cuda()

    with torch.no_grad():
        trt_output = trt_model({'input': x})['output']

    torch.testing.assert_allclose(trt_output, x + 1)

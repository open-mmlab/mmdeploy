import pytest
import torch

from mmdeploy.core import register_extra_symbolics
from mmdeploy.utils.test import WrapFunction
from .utils import TestOnnxRTExporter, TestTensorRTExporter

TEST_TENSORRT = TestTensorRTExporter()
TEST_ONNXRT = TestOnnxRTExporter()
ALL_BACKEND = [TEST_TENSORRT, TEST_ONNXRT]


@pytest.mark.parametrize('backend', ALL_BACKEND)
@pytest.mark.parametrize('pool_h,pool_w,spatial_scale,sampling_ratio',
                         [(2, 2, 1.0, 2), (4, 4, 2.0, 4)])
def test_roi_align(backend,
                   pool_h,
                   pool_w,
                   spatial_scale,
                   sampling_ratio,
                   inputs=None,
                   work_dir=None):
    backend.check_env()
    # using rewriter of roi_align to bypass mmcv has_custom_ops check.
    register_extra_symbolics(cfg=dict(), backend='default', opset=11)
    from mmcv.ops import roi_align

    def wrapped_function(torch_input, torch_rois):
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    wrapped_model = WrapFunction(wrapped_function)
    if not inputs:
        input = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
    else:
        input = torch.tensor(inputs[0], dtype=torch.float32)
        single_roi = torch.tensor(inputs[1], dtype=torch.float32)

    backend.run_and_validate(
        wrapped_model, [input, single_roi],
        'roi_align',
        input_names=['input', 'rois'],
        output_names=['roi_feat'],
        work_dir=work_dir)

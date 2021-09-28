import pytest
import torch
import torch.nn as nn

from mmdeploy.core import register_extra_symbolics
from mmdeploy.utils.test import WrapFunction
from .utils import TestNCNNExporter, TestOnnxRTExporter, TestTensorRTExporter

TEST_ONNXRT = TestOnnxRTExporter()
TEST_TENSORRT = TestTensorRTExporter()
TEST_NCNN = TestNCNNExporter()


@pytest.mark.parametrize('backend', [TEST_TENSORRT, TEST_ONNXRT])
@pytest.mark.parametrize('pool_h,pool_w,spatial_scale,sampling_ratio',
                         [(2, 2, 1.0, 2), (4, 4, 2.0, 4)])
def test_roi_align(backend,
                   pool_h,
                   pool_w,
                   spatial_scale,
                   sampling_ratio,
                   input_list=None,
                   save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
    else:
        input = torch.tensor(input_list[0], dtype=torch.float32)
        single_roi = torch.tensor(input_list[1], dtype=torch.float32)

    register_extra_symbolics(
        cfg=dict(), backend=backend.backend_name, opset=11)

    from mmcv.ops import roi_align

    def wrapped_function(torch_input, torch_rois):
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    wrapped_model = WrapFunction(wrapped_function).eval()

    backend.run_and_validate(
        wrapped_model, [input, single_roi],
        'roi_align',
        input_names=['input', 'rois'],
        output_names=['roi_feat'],
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT, TEST_ONNXRT])
@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
@pytest.mark.parametrize('padding_mode', ['zeros', 'border', 'reflection'])
@pytest.mark.parametrize('align_corners', [True, False])
def test_grid_sample(backend,
                     mode,
                     padding_mode,
                     align_corners,
                     input_list=None,
                     save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand(1, 1, 10, 10)
    else:
        input = torch.tensor(input_list[0])
    grid = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
    grid = nn.functional.affine_grid(
        grid, (1, 1, input.shape[2] * 2, input.shape[3] * 2)).type_as(input)

    register_extra_symbolics(
        cfg=dict(), backend=backend.backend_name, opset=11)

    def wrapped_function(inputs, grid):
        return nn.functional.grid_sample(
            inputs,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)

    wrapped_model = WrapFunction(wrapped_function).eval()

    backend.run_and_validate(
        wrapped_model, [input, grid],
        'grid_sampler',
        input_names=['input', 'grid'],
        output_names=['output'],
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT, TEST_ONNXRT])
@pytest.mark.parametrize('in_channels,out_channels,stride,padding,'
                         'dilation,groups,deform_groups,kernel_size',
                         [(3, 64, 1, 0, 1, 1, 1, 3),
                          (1, 32, 3, 2, 1, 1, 1, 3)])
@pytest.mark.parametrize('bias', [True, False])
def test_modulated_deform_conv(backend,
                               in_channels,
                               out_channels,
                               stride,
                               padding,
                               dilation,
                               groups,
                               deform_groups,
                               kernel_size,
                               bias,
                               input_list=None,
                               save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand(
            1, in_channels, 28, 28, requires_grad=False)  # (n, c, h, w)
    else:
        input = torch.tensor(input_list[0])
    conv_offset = nn.Conv2d(
        in_channels=in_channels,
        out_channels=deform_groups * 3 * kernel_size * kernel_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True)
    out = conv_offset(input)
    o1, o2, mask = torch.chunk(out, 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = torch.sigmoid(mask)

    from mmcv.ops import ModulatedDeformConv2d
    model = ModulatedDeformConv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups,
                                  deform_groups, bias).eval()

    backend.run_and_validate(
        model, [input, offset, mask],
        'modulated_deform_conv',
        input_names=['input', 'offset', 'mask'],
        output_names=['output'],
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('dynamic_export', [True, False])
@pytest.mark.parametrize('fp16_mode', [True, False])
@pytest.mark.parametrize('n, c, h, w', [(2, 3, 10, 10)])
def test_instance_norm(backend,
                       dynamic_export,
                       fp16_mode,
                       n,
                       c,
                       h,
                       w,
                       input_list=None,
                       save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.randn(n, c, h, w)
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'n',
                2: 'h',
                3: 'w',
            },
            'output': {
                0: 'n',
                2: 'h',
                3: 'w',
            },
        }
    else:
        dynamic_axes = None

    register_extra_symbolics(
        cfg=dict(), backend=backend.backend_name, opset=11)
    norm = nn.InstanceNorm2d(c, affine=True)
    wrapped_model = WrapFunction(norm).eval()

    backend.run_and_validate(
        wrapped_model, [input],
        'instance_norm',
        input_names=['input'],
        dynamic_axes=dynamic_axes,
        output_names=['output'],
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('k', [1, 3, 5])
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('largest', [True, False])
@pytest.mark.parametrize('sorted', [True, False])
def test_topk(backend,
              k,
              dim,
              largest,
              sorted,
              input_list=None,
              save_dir=None):
    backend.check_env()

    if not input_list:
        input = torch.rand(1, 8, 12, 17)
    assert input.shape[0] == 1, (f'ncnn batch must be 1, \
        but not {input.shape[0]}')
    cfg = dict()
    register_extra_symbolics(cfg=cfg, opset=11)

    def wrapped_function(inputs):
        return torch.Tensor.topk(inputs, k, dim, largest, sorted)

    wrapped_model = WrapFunction(wrapped_function)

    # when the 'sorted' attribute is False, pytorch will return
    # a hard to expect result, which only features that the topk
    # number is right. So the Topk unittest only check whether the
    # topk elements are right, all the possible order will be accepted.
    if not sorted:
        backend.run_and_validate(
            wrapped_model, [input.float()],
            'topk' + f'_no_sorted_dim_{dim}',
            input_names=['inputs'],
            output_names=['data', 'index'],
            save_dir=save_dir)
    else:
        backend.run_and_validate(
            wrapped_model, [input.float()],
            'topk',
            input_names=['inputs'],
            output_names=['data', 'index'],
            save_dir=save_dir)

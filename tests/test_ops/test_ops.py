# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import onnx
import pytest
import torch
import torch.nn as nn
from mmengine import Config
from onnx.helper import (make_graph, make_model, make_node,
                         make_tensor_value_info)

from mmdeploy.core import RewriterContext
from mmdeploy.utils.test import WrapFunction, assert_allclose
from .utils import TestNCNNExporter, TestOnnxRTExporter, TestTensorRTExporter

TEST_ONNXRT = TestOnnxRTExporter()
TEST_TENSORRT = TestTensorRTExporter()
TEST_NCNN = TestNCNNExporter()


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
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

    from mmcv.ops import roi_align

    def wrapped_function(torch_input, torch_rois):
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    wrapped_model = WrapFunction(wrapped_function).eval()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
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

    def wrapped_function(inputs, grid):
        return nn.functional.grid_sample(
            inputs,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)

    wrapped_model = WrapFunction(wrapped_function).eval()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            wrapped_model, [input, grid],
            'grid_sampler',
            input_names=['input', 'grid'],
            output_names=['output'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('dynamic_export', [True, False])
@pytest.mark.parametrize('mode', ['bicubic', 'nearest'])
@pytest.mark.parametrize('align_corners', [True, False])
@pytest.mark.parametrize('output_size', [[10, 20], None])
@pytest.mark.parametrize('scale_factor', [2])
@pytest.mark.parametrize('n, c, h, w', [(2, 3, 5, 10)])
def test_bicubic_interpolate(backend,
                             dynamic_export,
                             mode,
                             align_corners,
                             output_size,
                             scale_factor,
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

    if mode == 'nearest':
        align_corners = None
    if output_size is None:
        resize = nn.Upsample(
            scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    else:
        resize = nn.Upsample(
            size=output_size, mode=mode, align_corners=align_corners)
    expected_result = resize(input).cuda()
    wrapped_model = WrapFunction(resize).eval()

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [input],
            'bicubic_interpolate',
            input_names=['input'],
            dynamic_axes=dynamic_axes,
            output_names=['output'],
            save_dir=save_dir,
            expected_result=expected_result)


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

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            model, [input, offset, mask],
            'modulated_deform_conv',
            input_names=['input', 'offset', 'mask'],
            output_names=['output'],
            tolerate_small_mismatch=True,
            save_dir=save_dir)


@pytest.mark.parametrize('in_channels,out_channels,stride,padding,'
                         'dilation,groups,deform_groups,kernel_size',
                         [(1, 32, 3, 2, 1, 1, 1, 3)])
def test_deform_conv(in_channels, out_channels, stride, padding, dilation,
                     groups, deform_groups, kernel_size):

    inputs = torch.rand(
        1, in_channels, 28, 28, requires_grad=False)  # (n, c, h, w)

    conv_offset = nn.Conv2d(
        in_channels=in_channels,
        out_channels=deform_groups * 2 * kernel_size * kernel_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True)
    offsets = conv_offset(inputs)

    from mmcv.ops import DeformConv2d
    model = DeformConv2d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, deform_groups).eval()
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
    with RewriterContext(cfg={}, backend='tensorrt', opset=11):
        with torch.no_grad():
            torch.onnx.export(
                model, (inputs, offsets),
                onnx_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                opset_version=11)
        model = onnx.load(onnx_file)
        node = list(model.graph.node)[0]
        assert node.domain == 'mmdeploy'
        assert node.op_type == 'MMCVDeformConv2d'


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

    wrapped_model = nn.InstanceNorm2d(c, affine=True).eval().cuda()

    cudnn_enable = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [input],
            'instance_norm',
            input_names=['input'],
            dynamic_axes=dynamic_axes,
            output_names=['output'],
            save_dir=save_dir)
    torch.backends.cudnn.enabled = cudnn_enable


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('num_classes,pre_topk,after_topk,iou_threshold,'
                         'score_threshold,background_label_id',
                         [(5, 6, 3, 0.7, 0.1, -1)])
def test_batched_nms(backend,
                     num_classes,
                     pre_topk,
                     after_topk,
                     iou_threshold,
                     score_threshold,
                     background_label_id,
                     input_list=None,
                     save_dir=None):
    backend.check_env()

    if input_list is None:
        nms_boxes = torch.tensor([[[291.1746, 316.2263, 343.5029, 347.7312],
                                   [288.4846, 315.0447, 343.7267, 346.5630],
                                   [288.5307, 318.1989, 341.6425, 349.7222],
                                   [918.9102, 83.7463, 933.3920, 164.9041],
                                   [895.5786, 78.2361, 907.8049, 172.0883],
                                   [292.5816, 316.5563, 340.3462, 352.9989],
                                   [609.4592, 83.5447, 631.2532, 144.0749],
                                   [917.7308, 85.5870, 933.2839, 168.4530],
                                   [895.5138, 79.3596, 908.2865, 171.0418],
                                   [291.4747, 318.6987, 347.1208, 349.5754]]])
        scores = torch.tensor([[[0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
                                [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
                                [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
                                [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
                                [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
                                [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
                                [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
                                [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
                                [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
                                [0.4385, 0.6035, 0.0508, 0.0662, 0.5938]]])
    else:
        nms_boxes = torch.tensor(input_list[0], dtype=torch.float32)
        scores = torch.tensor(input_list[1], dtype=torch.float32)

    from mmdeploy.mmcv.ops.nms import _multiclass_nms
    expected_result = _multiclass_nms(
        nms_boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_topk + 1,
        keep_top_k=after_topk + 1)
    expected_result = (expected_result[0][:,
                                          0:-1, :], expected_result[1][:,
                                                                       0:-1])

    boxes = nms_boxes.unsqueeze(2).tile(num_classes, 1)

    from mmdeploy.mmcv.ops.nms import TRTBatchedNMSop
    batched_nms = TRTBatchedNMSop.apply

    def wrapped_function(boxes, scores):
        return batched_nms(boxes, scores, num_classes, pre_topk, after_topk,
                           iou_threshold, score_threshold, background_label_id)

    wrapped_model = WrapFunction(wrapped_function)

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [boxes, scores],
            'batched_nms',
            input_names=['boxes', 'scores'],
            output_names=['batched_nms_bboxes', 'inds'],
            expected_result=expected_result,
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('num_classes,pre_topk,after_topk,iou_threshold,'
                         'score_threshold,background_label_id',
                         [(5, 6, 3, 0.7, 0.1, -1)])
def test_batched_rotated_nms(backend,
                             num_classes,
                             pre_topk,
                             after_topk,
                             iou_threshold,
                             score_threshold,
                             background_label_id,
                             input_list=None,
                             save_dir=None):
    backend.check_env()
    pytest.importorskip('mmrotate', reason='mmrorate is not installed.')

    if input_list is None:
        nms_boxes = torch.tensor(
            [[[291.1746, 316.2263, 343.5029, 347.7312, 1.],
              [288.4846, 315.0447, 343.7267, 346.5630, 2.],
              [288.5307, 318.1989, 341.6425, 349.7222, 3.],
              [918.9102, 83.7463, 933.3920, 164.9041, 4.],
              [895.5786, 78.2361, 907.8049, 172.0883, 5.],
              [292.5816, 316.5563, 340.3462, 352.9989, 6.],
              [609.4592, 83.5447, 631.2532, 144.0749, 7.],
              [917.7308, 85.5870, 933.2839, 168.4530, 8.],
              [895.5138, 79.3596, 908.2865, 171.0418, 9.],
              [291.4747, 318.6987, 347.1208, 349.5754, 10.]]])
        scores = torch.tensor([[[0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
                                [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
                                [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
                                [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
                                [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
                                [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
                                [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
                                [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
                                [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
                                [0.4385, 0.6035, 0.0508, 0.0662, 0.5938]]])
    else:
        nms_boxes = torch.tensor(input_list[0], dtype=torch.float32)
        scores = torch.tensor(input_list[1], dtype=torch.float32)

    from mmdeploy.mmcv.ops.nms_rotated import _multiclass_nms_rotated
    expected_result = _multiclass_nms_rotated(
        nms_boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_topk + 1,
        keep_top_k=after_topk + 1)
    expected_result = (expected_result[0][:,
                                          0:-1, :], expected_result[1][:,
                                                                       0:-1])

    boxes = nms_boxes.unsqueeze(2).tile(num_classes, 1)

    from mmdeploy.mmcv.ops.nms_rotated import TRTBatchedRotatedNMSop
    batched_rotated_nms = TRTBatchedRotatedNMSop.apply

    def wrapped_function(boxes, scores):
        return batched_rotated_nms(boxes, scores, num_classes, pre_topk,
                                   after_topk, iou_threshold, score_threshold,
                                   background_label_id)

    wrapped_model = WrapFunction(wrapped_function)

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [boxes, scores],
            'batched_rotated_nms',
            input_names=['boxes', 'scores'],
            output_names=['batched_rotated_nms_bboxes', 'inds'],
            expected_result=expected_result,
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize(
    'out_size, pool_mode, sampling_ratio,roi_scale_factor,'
    ' finest_scale,featmap_strides, aligned',
    [(tuple([2, 2]), 0, 2, 1.0, 2, list([2.0, 4.0]), 1),
     (tuple([2, 2]), 1, 2, 1.0, 2, list([2.0, 4.0]), 1)])
def test_multi_level_roi_align(backend,
                               out_size,
                               pool_mode,
                               sampling_ratio,
                               roi_scale_factor,
                               finest_scale,
                               featmap_strides,
                               aligned,
                               input_list=None,
                               save_dir=None):
    backend.check_env()

    if input_list is None:
        input = [
            torch.tensor([[[[0.3014, 0.7334, 0.6502, 0.1689],
                            [0.3031, 0.3735, 0.6032, 0.1644],
                            [0.0393, 0.4415, 0.3858, 0.2657],
                            [0.5766, 0.0211, 0.6384, 0.0016]],
                           [[0.0811, 0.6255, 0.0247, 0.3471],
                            [0.1390, 0.9298, 0.6178, 0.6636],
                            [0.2243, 0.2024, 0.2366, 0.3660],
                            [0.1050, 0.2301, 0.7489, 0.7506]],
                           [[0.3868, 0.1706, 0.2390, 0.8494],
                            [0.2643, 0.9347, 0.0412, 0.5790],
                            [0.6202, 0.0682, 0.0390, 0.5296],
                            [0.5383, 0.1221, 0.6344, 0.1514]]]]),
            torch.tensor([[[[0.1939, 0.9983, 0.4031, 0.2712],
                            [0.7929, 0.1504, 0.0946, 0.5030],
                            [0.1421, 0.7908, 0.9595, 0.4198],
                            [0.6880, 0.4722, 0.9896, 0.2266]],
                           [[0.0778, 0.4232, 0.0736, 0.0168],
                            [0.2887, 0.8461, 0.1140, 0.9582],
                            [0.5169, 0.4924, 0.8275, 0.5530],
                            [0.8961, 0.7466, 0.5976, 0.3760]],
                           [[0.1542, 0.5028, 0.8412, 0.6617],
                            [0.3751, 0.2798, 0.3835, 0.8640],
                            [0.5821, 0.6588, 0.1324, 0.7619],
                            [0.9178, 0.7282, 0.0291, 0.3028]]]])
        ]
        rois = torch.tensor([[0., 0., 0., 4., 4.]])
        if pool_mode == 1:
            expected_result = torch.tensor([[[[0.1939, 0.3950],
                                              [0.3437, 0.4543]],
                                             [[0.0778, 0.1641],
                                              [0.1305, 0.2301]],
                                             [[0.1542, 0.2413],
                                              [0.2094, 0.2688]]]])
        else:
            expected_result = torch.tensor([[[[0.1939, 0.4956],
                                              [0.4185, 0.5167]],
                                             [[0.0778, 0.2073],
                                              [0.1569, 0.3162]],
                                             [[0.1542, 0.2849],
                                              [0.2370, 0.3053]]]])

    else:
        input = input_list[0]
        rois = input_list[1]
        expected_result = input_list[2]
    input_name = [('input_' + str(i)) for i in range(len(featmap_strides))]
    input_name.insert(0, 'rois')

    inputs = [
        onnx.helper.make_tensor_value_info(
            input_name[i + 1], onnx.TensorProto.FLOAT, shape=input[i].shape)
        for i in range(len(input_name) - 1)
    ]
    inputs.append(
        onnx.helper.make_tensor_value_info(
            'rois', onnx.TensorProto.FLOAT, shape=rois.shape))
    outputs = [
        onnx.helper.make_tensor_value_info(
            'bbox_feats', onnx.TensorProto.FLOAT, shape=expected_result.shape)
    ]
    node = onnx.helper.make_node(
        'MMCVMultiLevelRoiAlign',
        input_name, ['bbox_feats'],
        'MMCVMultiLevelRoiAlign_0',
        None,
        'mmdeploy',
        pool_mode=pool_mode,
        aligned=aligned,
        featmap_strides=featmap_strides,
        finest_scale=finest_scale,
        output_height=out_size[0],
        output_width=out_size[1],
        roi_scale_factor=roi_scale_factor,
        sampling_ratio=sampling_ratio)
    graph = onnx.helper.make_graph([node], 'torch-jit-export', inputs, outputs)
    onnx_model = onnx.helper.make_model(
        graph, producer_name='pytorch', producer_version='1.8')
    onnx_model.opset_import[0].version = 11
    onnx_model.opset_import.append(
        onnx.onnx_ml_pb2.OperatorSetIdProto(domain='mmdeploy', version=1))

    backend.run_and_validate(
        onnx_model, [rois, *input],
        'multi_level_roi_align',
        input_names=input_name,
        output_names=['bbox_feats'],
        expected_result=expected_result,
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

    if input_list is None:
        input = torch.rand(1, 8, 12, 17)
    else:
        input = input_list[0]
    assert input.shape[0] == 1, (f'ncnn batch must be 1, \
        but got {input.shape[0]}')

    def topk_function(inputs):
        return torch.Tensor.topk(inputs, k, dim, largest, sorted)

    wrapped_model = WrapFunction(topk_function)

    # when the 'sorted' attribute is False, pytorch will return
    # a hard to expect result, which only features that the topk
    # number is right. So the Topk unittest only check whether the
    # topk elements are right, all the possible order will be accepted.
    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
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


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('dim, n, c, h, w', [(1, 1, 1, 1, 8), (2, 1, 1, 5, 7),
                                             (3, 1, 3, 10, 15)])
def test_shape(backend,
               dim,
               n,
               c,
               h,
               w,
               input_names=['input'],
               output_names=['output'],
               tolerate_small_mismatch=False,
               input_list=None,
               save_dir=None):
    backend.check_env()

    orig_shape = (n, c, h, w)[-dim - 1:]
    if input_list is None:
        input = torch.rand(orig_shape)
    else:
        input = input_list[0]
        assert input.dim() == dim + 1, 'input.dim() must equal to dim + 1'
        assert tuple(input.shape) == orig_shape, 'input.shape must the \
            same as orig_shape'

    assert input.shape[0] == 1, (f'ncnn batch must be 1, \
        but got {input.shape[0]}')

    shape_node = make_node('Shape', input_names, output_names)
    assert len(input_names) == 1, 'length of input_names must be 1'
    assert len(output_names) == 1, 'length of output_names must be 1'
    shape_graph = make_graph([shape_node], 'shape_graph', [
        make_tensor_value_info(input_names[0], onnx.TensorProto.FLOAT,
                               orig_shape)
    ], [
        make_tensor_value_info(output_names[0], onnx.TensorProto.FLOAT,
                               (dim + 1, ))
    ])
    shape_model = make_model(shape_graph)

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        ncnn_model = backend.onnx2ncnn(shape_model, 'shape', output_names,
                                       save_dir)

    # ncnn mat has implicit batch for mat, the ncnn_output is a mat,
    # so the ncnn_outputs has 2 dimensions, not 1.
    model_outputs = [torch.tensor(orig_shape).unsqueeze(0).float()]
    ncnn_outputs = ncnn_model(dict(zip(input_names, [input])))
    ncnn_outputs = [ncnn_outputs[name] for name in output_names]
    assert_allclose(model_outputs, ncnn_outputs, tolerate_small_mismatch)


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('dim, n, c, h, w', [(1, 1, 1, 1, 8), (2, 1, 1, 5, 7),
                                             (3, 1, 3, 10, 15)])
@pytest.mark.parametrize('val', [0., 1., -3, 4.25])
def test_constantofshape(backend,
                         dim,
                         n,
                         c,
                         h,
                         w,
                         val,
                         input_names=['input'],
                         output_names=['output'],
                         tolerate_small_mismatch=False,
                         input_list=None,
                         save_dir=None):
    backend.check_env()
    if input_list is None:
        input = torch.tensor((n, c, h, w)[-dim - 1:]).unsqueeze(0)
    else:
        input = input_list[0]
        assert input.dim() == dim + 1, 'input.dim() must equal to dim + 1'
        assert tuple(input.shape) == (n, c, h,
                                      w)[-dim - 1:], 'input.shape must the \
            same as orig_shape'

    assert input.shape[0] == 1, (f'ncnn input batch must be 1, \
        got {input.shape[0]}')
    assert input[0][0] == 1, (f'ncnn output mat batch must be 1, \
        got {input[0][0]}')

    constantofshape_node = make_node(
        'ConstantOfShape', input_names, output_names, value=float(val))
    assert len(input_names) == 1, 'length of input_names must be 1'
    assert len(output_names) == 1, 'length of output_names must be 1'
    constantofshape_graph = make_graph(
        [constantofshape_node], 'constantofshape_graph', [
            make_tensor_value_info(input_names[0], onnx.TensorProto.FLOAT,
                                   input.shape)
        ], [
            make_tensor_value_info(output_names[0], onnx.TensorProto.FLOAT,
                                   torch.Size(input[0]))
        ])
    constantofshape_model = make_model(constantofshape_graph)
    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        ncnn_model = backend.onnx2ncnn(constantofshape_model,
                                       'constantofshape', output_names,
                                       save_dir)

    # ncnn mat has implicit batch for mat, the ncnn_output is a mat,
    # so the ncnn_outputs has 2 dimensions, not 1.
    model_outputs = [torch.fill_(torch.rand(tuple(input[0])), val)]
    ncnn_outputs = ncnn_model(dict(zip(input_names, [input.float()])))
    ncnn_outputs = [ncnn_outputs[name] for name in output_names]
    assert_allclose(model_outputs, ncnn_outputs, tolerate_small_mismatch)


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('axis, data_dims, indice_dims', [(0, 1, 1), (0, 2, 1),
                                                          (1, 2, 1), (0, 3, 1),
                                                          (1, 3, 1),
                                                          (2, 3, 1)])
def test_gather(backend,
                axis,
                data_dims,
                indice_dims,
                input_names=['input', 'indices'],
                output_names=['output'],
                tolerate_small_mismatch=False,
                input_list=None,
                save_dir=None):
    backend.check_env()

    if input_list is None:
        # the real data dims is data_dims + 1
        data = torch.rand((8, 12, 17)[-data_dims:]).unsqueeze(0)
        indice = torch.randint(0, 8, (3, 4, 5)[-indice_dims:]).unsqueeze(0)
    else:
        data = input_list[0]
        indice = input_list[1]
    assert data.shape[0] == 1, ('ncnn batch must be 1,'
                                f'but got {data.shape[0]}')
    assert indice.shape[0] == 1, ('ncnn batch must be 1,'
                                  f'but got {indice.shape[0]}')

    gather_node = make_node('Gather', input_names, output_names, axis=axis + 1)
    gather_graph = make_graph([gather_node], 'gather_graph', [
        make_tensor_value_info(input_names[0], onnx.TensorProto.FLOAT, None),
        make_tensor_value_info(input_names[1], onnx.TensorProto.INT64, None)
    ], [make_tensor_value_info(output_names[0], onnx.TensorProto.FLOAT, None)])
    opset_imports = [onnx.helper.make_operatorsetid('', 11)]
    gather_model = make_model(gather_graph, opset_imports=opset_imports)
    gather_model.ir_version = 7
    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        ncnn_model = backend.onnx2ncnn(gather_model, 'gather', output_names,
                                       save_dir)

    # ncnn mat has implicit batch for mat, the ncnn_output is a mat,
    # so the ncnn_outputs has 2 dimensions, not 1.
    import importlib

    assert importlib.util.find_spec('onnxruntime') is not None, 'onnxruntime \
         not installed.'

    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_model = ORTWrapper(
        gather_model.SerializeToString(),
        device='cpu',
        output_names=output_names)
    model_outputs = ort_model(dict(zip(input_names, [data, indice[0]])))
    model_outputs = ort_model.output_to_list(model_outputs)

    ncnn_outputs = ncnn_model(
        dict(zip(input_names, [data.float(), indice.float()])))
    ncnn_outputs = [ncnn_outputs[name] for name in output_names]
    assert_allclose(model_outputs, ncnn_outputs, tolerate_small_mismatch)


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_tensorslice(backend, dim, input_list=None, save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand((8, 12, 17)[-dim:]).unsqueeze(0)
    else:
        input = input_list[0]
        assert input.dim() == dim + 1, f'input.dim() must equal to \
            dim + 1, expected: {dim + 1}, got: {input.dim()}'

    assert input.shape[0] == 1, (f'ncnn batch must be 1, \
        but got {input.shape[0]}')

    def tensorslice_function(inputs):
        if dim == 1:
            return inputs[:, 2:17:7]
        if dim == 2:
            return inputs[:, 3:12:4, 2:15:3]
        if dim == 3:
            return inputs[:, 0:8:2, 2:12:4, 2:17:7]

    wrapped_model = WrapFunction(tensorslice_function)

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [input.float()],
            'tensorslice',
            input_names=['inputs'],
            output_names=['outputs'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_NCNN])
@pytest.mark.parametrize('input_dim, output_dim', [(1, 1), (1, 2), (1, 3),
                                                   (2, 2), (2, 3), (3, 3)])
def test_expand(backend,
                input_dim,
                output_dim,
                input_list=None,
                save_dir=None):
    backend.check_env()
    if input_list is None:
        input = torch.rand((1, 12, 1)[-input_dim:]).unsqueeze(0)
        target = torch.rand((8, 12, 17)[-output_dim:]).unsqueeze(0)
    else:
        input = input_list[0]
        target = input_list[1]
    assert input.shape[0] == 1, (f'ncnn batch must be 1, \
        but not {input.shape[0]}')
    assert target.shape[0] == 1, (f'ncnn batch must be 1, \
        but not {target.shape[0]}')

    def expand_function(input, target):
        return input.expand_as(target)

    wrapped_model = WrapFunction(expand_function)
    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [input.float(), target.float()],
            'expand',
            input_names=['input', 'shape'],
            output_names=['output'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_ONNXRT])
@pytest.mark.parametrize('iou_threshold', [0.1, 0.3])
@pytest.mark.parametrize('score_threshold', [0., 0.1])
def test_nms_rotated(backend, iou_threshold, score_threshold, save_dir=None):
    backend.check_env()

    boxes = torch.tensor(
        [[[60, 75, 20, 50, 0], [65, 80, 10, 40, 0], [30, 30, 40, 40, 0]],
         [[60, 75, 20, 50, 0], [65, 80, 10, 40, 0], [30, 30, 40, 40, 0]]],
        dtype=torch.float32)
    scores = torch.tensor(
        [[[0.5, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.7], [0.1, 0.1, 0.1]],
         [[0.1, 0.1, 0.1], [0.7, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.5]]],
        dtype=torch.float32)

    from mmdeploy.mmcv.ops import ONNXNMSRotatedOp

    def wrapped_function(torch_boxes, torch_scores):
        return ONNXNMSRotatedOp.apply(torch_boxes, torch_scores, iou_threshold,
                                      score_threshold)

    wrapped_model = WrapFunction(wrapped_function).eval()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            wrapped_model, [boxes, scores],
            'nms_rotated',
            input_names=['boxes', 'scores'],
            output_names=['keep_inds'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_ONNXRT])
@pytest.mark.parametrize('pool_h,pool_w,spatial_scale,sampling_ratio',
                         [(2, 2, 1.0, 2), (4, 4, 2.0, 4)])
def test_roi_align_rotated(backend,
                           pool_h,
                           pool_w,
                           spatial_scale,
                           sampling_ratio,
                           input_list=None,
                           save_dir=None):
    backend.check_env()

    if input_list is None:
        # input = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        input = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float32)
        single_roi = torch.tensor([[0., 0.5, 0.5, 1., 1., 0]],
                                  dtype=torch.float32)
    else:
        input = torch.tensor(input_list[0], dtype=torch.float32)
        single_roi = torch.tensor(input_list[1], dtype=torch.float32)

    from mmcv.ops import roi_align_rotated

    def wrapped_function(torch_input, torch_rois):
        return roi_align_rotated(torch_input, torch_rois, (pool_w, pool_h),
                                 spatial_scale, sampling_ratio, True, False)

    wrapped_model = WrapFunction(wrapped_function).eval()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            wrapped_model, [input, single_roi],
            'roi_align_rotated',
            input_names=['input', 'rois'],
            output_names=['roi_feat'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize(
    'out_size, clockwise, sampling_ratio, roi_scale_factor,'
    ' finest_scale, featmap_strides, aligned',
    [(tuple([2, 2]), False, 2, 1.0, 2, list([1.0]), 1)])
def test_multi_level_rotated_roi_align(backend,
                                       out_size,
                                       clockwise,
                                       sampling_ratio,
                                       roi_scale_factor,
                                       finest_scale,
                                       featmap_strides,
                                       aligned,
                                       input_list=None,
                                       save_dir=None):
    backend.check_env()

    if input_list is None:
        import numpy as np
        input = [
            torch.tensor([[[[1., 2., 5., 6.], [3., 4., 7., 8.],
                            [9., 10., 13., 14.], [11., 12., 15., 16.]]]])
        ]
        rois = torch.tensor([[0., 1.5, 1.5, 3., 3., np.pi / 2]])
        expected_result = torch.tensor([[[[7.5625, 1.9375], [10.375, 4.75]]]])
    else:
        input = input_list[0]
        rois = input_list[1]
        expected_result = input_list[2]
    input_name = [('input_' + str(i)) for i in range(len(featmap_strides))]
    input_name.insert(0, 'rois')

    inputs = [
        onnx.helper.make_tensor_value_info(
            input_name[i + 1], onnx.TensorProto.FLOAT, shape=input[i].shape)
        for i in range(len(input_name) - 1)
    ]
    inputs.append(
        onnx.helper.make_tensor_value_info(
            'rois', onnx.TensorProto.FLOAT, shape=rois.shape))
    outputs = [
        onnx.helper.make_tensor_value_info(
            'bbox_feats', onnx.TensorProto.FLOAT, shape=expected_result.shape)
    ]
    node = onnx.helper.make_node(
        'MMCVMultiLevelRotatedRoiAlign',
        input_name, ['bbox_feats'],
        'MMCVMultiLevelRotatedRoiAlign_0',
        None,
        'mmdeploy',
        featmap_strides=featmap_strides,
        finest_scale=finest_scale,
        output_height=out_size[0],
        output_width=out_size[1],
        clockwise=clockwise,
        roi_scale_factor=roi_scale_factor,
        sampling_ratio=sampling_ratio,
        aligned=aligned)
    graph = onnx.helper.make_graph([node], 'torch-jit-export', inputs, outputs)
    onnx_model = onnx.helper.make_model(
        graph, producer_name='pytorch', producer_version='1.8')
    onnx_model.opset_import[0].version = 11
    onnx_model.opset_import.append(
        onnx.onnx_ml_pb2.OperatorSetIdProto(domain='mmdeploy', version=1))

    backend.run_and_validate(
        onnx_model, [rois, *input],
        'multi_level_rotated_roi_align',
        input_names=input_name,
        output_names=['bbox_feats'],
        expected_result=expected_result,
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('strides', [(4, 4)])
def test_trt_grid_priors(backend, strides, input_list=None, save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand(1, 3, 2, 2)
        base_anchors = torch.tensor([[-22.6274, -11.3137, 22.6274, 11.3137],
                                     [-16.0000, -16.0000, 16.0000, 16.0000],
                                     [-11.3137, -22.6274, 11.3137, 22.6274]])

        expected_result = torch.tensor([[-22.6274, -11.3137, 22.6274, 11.3137],
                                        [-16.0000, -16.0000, 16.0000, 16.0000],
                                        [-11.3137, -22.6274, 11.3137, 22.6274],
                                        [-18.6274, -11.3137, 26.6274, 11.3137],
                                        [-12.0000, -16.0000, 20.0000, 16.0000],
                                        [-7.3137, -22.6274, 15.3137, 22.6274],
                                        [-22.6274, -7.3137, 22.6274, 15.3137],
                                        [-16.0000, -12.0000, 16.0000, 20.0000],
                                        [-11.3137, -18.6274, 11.3137, 26.6274],
                                        [-18.6274, -7.3137, 26.6274, 15.3137],
                                        [-12.0000, -12.0000, 20.0000, 20.0000],
                                        [-7.3137, -18.6274, 15.3137, 26.6274]])
    else:
        input = input_list[0]
        base_anchors = input_list[1]
        expected_result = input_list[2]
    input_name = ['input']
    output_name = ['output']

    class GridPriorsTestOps(torch.autograd.Function):

        @staticmethod
        def forward(ctx, base_anchor, feat_h, feat_w, stride_h: int,
                    stride_w: int):
            a = base_anchor.shape[0]
            return base_anchor.new_empty(feat_h * feat_w * a, 4)

        @staticmethod
        def symbolic(g, base_anchor, feat_h, feat_w, stride_h: int,
                     stride_w: int):
            from torch.onnx import symbolic_helper
            feat_h = symbolic_helper._unsqueeze_helper(g, feat_h, [0])
            feat_w = symbolic_helper._unsqueeze_helper(g, feat_w, [0])
            zero_h = g.op(
                'ConstantOfShape',
                feat_h,
                value_t=torch.tensor([0], dtype=torch.long),
            )
            zero_w = g.op(
                'ConstantOfShape',
                feat_w,
                value_t=torch.tensor([0], dtype=torch.long),
            )
            return g.op(
                'mmdeploy::GridPriorsTRT',
                base_anchor,
                zero_h,
                zero_w,
                stride_h_i=stride_h,
                stride_w_i=stride_w)

    class GridPriorsTestModel(torch.nn.Module):

        def __init__(self, strides, base_anchors=base_anchors) -> None:
            super().__init__()
            self.strides = strides
            self.base_anchors = base_anchors

        def forward(self, x):
            base_anchors = self.base_anchors
            h, w = x.shape[2:]
            strides = self.strides
            return GridPriorsTestOps.apply(base_anchors, h, w, strides[0],
                                           strides[1])

    model = GridPriorsTestModel(strides=strides)

    backend.run_and_validate(
        model, [input],
        'trt_grid_priors',
        input_names=input_name,
        output_names=output_name,
        expected_result=expected_result,
        dynamic_axes=dict(input={
            2: 'h',
            3: 'w'
        }),
        save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
def test_dot_product_attention(backend, save_dir=None):
    backend.check_env()

    B = 2
    Nt = 4
    Ns = 4
    E = 2
    query = torch.rand(B, Nt, E).cuda()
    key = torch.rand(B, Ns, E).cuda()
    value = torch.rand(B, Ns, E).cuda()

    model = torch.nn.MultiheadAttention(E, 2).cuda()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            model, [query, key, value],
            'dot_product_attention',
            input_names=['query', 'key', 'value'],
            output_names=['out', 'attn'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
def test_gather_topk(backend, save_dir=None):
    backend.check_env()
    from mmdeploy.codebase.mmdet.deploy.utils import gather_topk

    x = torch.rand(2, 10, 4).cuda()

    class TestModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            batch_size = x.size(0)
            max_x, _ = x.max(-1)
            _, inds = max_x.topk(4)

            new_x = gather_topk(x, inds=inds, batch_size=batch_size)
            return new_x

    model = TestModel().cuda()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            model, [x],
            'gather_topk',
            input_names=['x'],
            output_names=['out'],
            save_dir=save_dir)


@pytest.mark.parametrize('backend', [TEST_ONNXRT])
@pytest.mark.parametrize('pre_top_k', [-1, 1000])
def test_multiclass_nms_rotated_with_keep_top_k(backend, pre_top_k):
    backend.check_env()
    from mmdeploy.mmcv.ops import multiclass_nms
    from mmdeploy.utils.test import get_onnx_model
    keep_top_k = 15
    deploy_cfg = Config(
        dict(
            onnx_config=dict(
                output_names=None,
                input_shape=None,
                dynamic_axes=dict(
                    boxes={
                        0: 'batch_size',
                        1: 'num_boxes'
                    },
                    scores={
                        0: 'batch_size',
                        1: 'num_boxes',
                        2: 'num_classes'
                    },
                ),
            ),
            backend_config=dict(type=backend.backend_name),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    pre_top_k=pre_top_k,
                    keep_top_k=keep_top_k,
                ))))

    num_classes = 5
    num_boxes = 2
    batch_size = 1
    export_boxes = torch.rand(batch_size, num_boxes, 5)
    export_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': export_boxes, 'scores': export_scores}

    wrapped_func = WrapFunction(
        multiclass_nms, nms_type='nms_rotated', keep_top_k=keep_top_k)

    onnx_model_path = get_onnx_model(
        wrapped_func, model_inputs=model_inputs, deploy_cfg=deploy_cfg)

    num_boxes = 100
    test_boxes = torch.rand(batch_size, num_boxes, 5)
    test_scores = torch.ones(batch_size, num_boxes, num_classes)
    model_inputs = {'boxes': test_boxes, 'scores': test_scores}

    import mmdeploy.backend.onnxruntime as ort_apis
    backend_model = ort_apis.ORTWrapper(onnx_model_path, 'cpu', None)
    output = backend_model.forward(model_inputs)
    output = backend_model.output_to_list(output)
    dets = output[0]

    # Subtract 1 dim since we pad the tensors
    assert dets.shape[1] - 1 < keep_top_k, \
        'multiclass_nms_rotated returned more values than "keep_top_k"\n' \
        f'dets.shape: {dets.shape}\n' \
        f'keep_top_k: {keep_top_k}'


@pytest.mark.parametrize('backend', [TEST_TENSORRT])
def test_multi_scale_deformable_attn(backend, save_dir=None):
    backend.check_env()
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttnFunction

    Bs = 2
    Nh = 8
    Nc = 32
    Nq = 32
    Np = 32
    spatial_shapes = [[68, 120], [34, 60]]
    value_spatial_shapes = torch.LongTensor(spatial_shapes).cuda()
    Nl = value_spatial_shapes.shape[0]
    Nk = sum([spatial_shapes[i][0] * spatial_shapes[i][1] for i in range(Nl)])
    value = torch.rand(Bs, Nk, Nh, Nc).cuda()
    level_start_index = torch.cat((
        value_spatial_shapes.new_zeros((1, )),
        value_spatial_shapes.prod(1).cumsum(0)[:-1].to(torch.int64),
    ))
    sampling_locations = torch.rand(Bs, Nq, Nh, Nl, Np, 2).cuda()
    attention_weights = torch.rand(Bs, Nq, Nh, Nl, Np).cuda()

    class TestModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.im2col_step = 32

        def forward(self, value, value_spatial_shapes, level_start_index,
                    sampling_locations, attention_weights):

            new_x = MultiScaleDeformableAttnFunction.apply(
                value, value_spatial_shapes, level_start_index,
                sampling_locations, attention_weights, self.im2col_step)
            return new_x

    model = TestModel().cuda()

    with RewriterContext(
            Config({'backend_config': {
                'type': backend.backend_name
            }}),
            backend=backend.backend_name,
            opset=11):
        backend.run_and_validate(
            model, [
                value, value_spatial_shapes, level_start_index,
                sampling_locations, attention_weights
            ],
            'multi_scale_deformable_attn',
            input_names=[
                'value', 'value_spatial_shapes', 'level_start_index',
                'sampling_locations', 'attention_weights'
            ],
            output_names=['output'],
            save_dir=save_dir)

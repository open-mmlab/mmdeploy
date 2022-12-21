# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import onnx
import pytest
import torch

from mmdeploy.core import RewriterContext
from mmdeploy.utils import Backend
from mmdeploy.utils.test import WrapFunction, check_backend


@pytest.mark.parametrize(
    'iou_threshold, score_threshold,max_output_boxes_per_class',
    [(0.6, 0.2, 3)])
def test_ONNXNMSop(iou_threshold, score_threshold, max_output_boxes_per_class):
    boxes = torch.tensor([[[291.1746, 316.2263, 343.5029, 347.7312],
                           [288.4846, 315.0447, 343.7267, 346.5630],
                           [288.5307, 318.1989, 341.6425, 349.7222],
                           [918.9102, 83.7463, 933.3920, 164.9041],
                           [895.5786, 78.2361, 907.8049, 172.0883],
                           [292.5816, 316.5563, 340.3462, 352.9989],
                           [609.4592, 83.5447, 631.2532, 144.0749],
                           [917.7308, 85.5870, 933.2839, 168.4530],
                           [895.5138, 79.3596, 908.2865, 171.0418],
                           [291.4747, 318.6987, 347.1208, 349.5754]]])
    scores = torch.rand(1, 5, 10)

    from mmdeploy.mmcv.ops import ONNXNMSop

    def wrapped_function(torch_bboxes, torch_scores):
        return ONNXNMSop.apply(torch_bboxes, torch_scores,
                               max_output_boxes_per_class, iou_threshold,
                               score_threshold)

    wrapped_model = WrapFunction(wrapped_function).eval()
    result = wrapped_model(boxes, scores)
    assert result is not None
    onnx_file_path = tempfile.NamedTemporaryFile().name
    with RewriterContext({}, opset=11), torch.no_grad():
        torch.onnx.export(
            wrapped_model, (boxes, scores),
            onnx_file_path,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['boxes', 'scores'],
            output_names=['result'],
            opset_version=11)
    model = onnx.load(onnx_file_path)
    assert model.graph.node[3].op_type == 'NonMaxSuppression'


def test_deform_conv_openvino():
    check_backend(Backend.OPENVINO)
    input = torch.Tensor([[[[1., 2., 3.], [0., 1., 2.], [3., 5., 2.]]]])
    offset = torch.Tensor([[[[1.7000, 2.9000], [3.4000, 4.8000]],
                            [[1.1000, 2.0000], [2.1000, 1.9000]],
                            [[3.1000, 5.1000], [5.9000, 4.9000]],
                            [[2.0000, 4.1000], [4.0000, 6.6000]],
                            [[1.6000, 2.7000], [3.8000, 3.1000]],
                            [[2.5000, 4.3000], [4.2000, 5.3000]],
                            [[1.7000, 3.3000], [3.6000, 4.5000]],
                            [[1.7000, 3.4000], [5.2000, 6.1000]]]])
    expected_output = torch.Tensor([[[[1.6500, 0.0000], [0.0000, 0.0000]]]])
    from mmcv.ops.deform_conv import DeformConv2dFunction

    def wrapped_function(input, offset):
        weight = torch.Tensor([[[[0.4000, 0.2000], [0.1000, 0.9000]]]])
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        deform_groups = 1
        return DeformConv2dFunction.apply(input, offset, weight, stride,
                                          padding, dilation, groups,
                                          deform_groups)

    wrapped_model = WrapFunction(wrapped_function).eval()

    model_output = wrapped_model(input, offset)

    assert torch.allclose(expected_output, model_output)
    onnx_file_path = tempfile.NamedTemporaryFile().name
    with RewriterContext({}, backend='openvino'), torch.no_grad():
        torch.onnx.export(
            wrapped_model, (input, offset),
            onnx_file_path,
            export_params=True,
            keep_initializers_as_inputs=True,
            input_names=['input', 'offset'],
            output_names=['result'],
            opset_version=11)
    model = onnx.load(onnx_file_path)
    assert model.graph.node[1].op_type == 'DeformableConv2D'
    assert model.graph.node[1].domain == 'org.openvinotoolkit'


def test_patch_embed_ncnn():
    check_backend(Backend.NCNN)

    from mmcv.cnn.bricks.transformer import PatchEmbed

    input = torch.ones((1, 3, 384, 384))
    patch_cfg = {
        'in_channels': 3,
        'input_size': 384,
        'embed_dims': 768,
        'conv_type': 'Conv2d',
        'kernel_size': 32,
        'stride': 32
    }
    wrapped_model = PatchEmbed(**patch_cfg)
    wrapped_model.eval()
    with RewriterContext({}, backend='ncnn'), torch.no_grad():
        _, shape = wrapped_model(input)
        assert shape[0] == patch_cfg['input_size'] / patch_cfg['stride']


def test_modulated_deform_conv():
    check_backend(Backend.TORCHSCRIPT)
    from mmdeploy.backend.torchscript import ops_available

    if not ops_available():
        pytest.skip('torchscript custom ops is required.')

    from mmcv.ops import ModulatedDeformConv2dPack

    from mmdeploy.apis.torch_jit import trace

    model = ModulatedDeformConv2dPack(3, 1, 1).eval()
    x = torch.rand(1, 3, 16, 16)

    jit_model = trace(model, x, None, backend='torchscript')

    out = model(x)
    jit_out = jit_model(x)

    torch.testing.assert_allclose(out, jit_out)

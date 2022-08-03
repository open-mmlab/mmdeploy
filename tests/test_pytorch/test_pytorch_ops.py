# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import onnx
import pytest
import torch
from mmcv import Config

from mmdeploy.core import RewriterContext

onnx_file = tempfile.NamedTemporaryFile(suffix='onnx').name


@pytest.fixture(autouse=False, scope='function')
def prepare_symbolics():
    context = RewriterContext(
        Config(
            dict(
                onnx_config=dict(
                    type='onnx',
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    opset_version=11,
                    save_file='end2end.onnx',
                    input_names=['input'],
                    output_names=['output'],
                    input_shape=None),
                backend_config=dict(type='tensorrt'))),
        'tensorrt',
        opset=11)
    context.enter()

    yield

    context.exit()


@pytest.fixture(autouse=False, scope='function')
def prepare_symbolics_ncnn():
    context = RewriterContext(
        Config({'backend_config': {
            'type': 'ncnn'
        }}), 'ncnn', opset=11)
    context.enter()

    yield

    context.exit()


class OpModel(torch.nn.Module):

    def __init__(self, func, *args):
        super().__init__()
        self._func = func
        self._arg_tuple = args

    def forward(self, x):
        return self._func(x, *self._arg_tuple)


def get_model_onnx_nodes(model, x, onnx_file=onnx_file):
    torch.onnx.export(model, x, onnx_file, opset_version=11)
    onnx_model = onnx.load(onnx_file)
    nodes = onnx_model.graph.node
    return nodes


@pytest.mark.usefixtures('prepare_symbolics')
class TestAdaptivePool:

    def test_adaptive_pool_2d_global(self):
        x = torch.rand(2, 2, 2)
        model = OpModel(torch.nn.functional.adaptive_avg_pool2d, [1, 1]).eval()
        nodes = get_model_onnx_nodes(model, x)
        assert nodes[0].op_type == 'GlobalAveragePool'

    def test_adaptive_pool_2d(self):
        x = torch.rand(2, 2, 2)
        model = OpModel(torch.nn.functional.adaptive_avg_pool2d, [2, 2]).eval()
        nodes = get_model_onnx_nodes(model, x)
        assert nodes[-1].op_type == 'AveragePool'


@pytest.mark.usefixtures('prepare_symbolics_ncnn')
def test_adaptive_pool_2d_ncnn():
    x = torch.rand(2, 2, 2)
    model = OpModel(torch.nn.functional.adaptive_avg_pool2d,
                    torch.tensor([2, 2], dtype=torch.int64)).eval()
    nodes = get_model_onnx_nodes(model, x)
    assert nodes[1].op_type == 'AdaptiveAvgPool2d'
    assert nodes[1].domain == 'mmdeploy'


@pytest.mark.usefixtures('prepare_symbolics')
def test_grid_sampler():
    x = torch.rand(1, 1, 2, 2)
    flow = torch.zeros([1, 2, 2, 2])
    model = OpModel(torch.grid_sampler, flow, 0, 0, False).eval()
    nodes = get_model_onnx_nodes(model, x)
    assert nodes[1].op_type == 'grid_sampler'
    assert nodes[1].domain == 'mmdeploy'


@pytest.mark.usefixtures('prepare_symbolics')
def test_instance_norm():
    x = torch.rand(1, 2, 2, 2)
    model = OpModel(torch.group_norm, 1, torch.rand([2]), torch.rand([2]),
                    1e-05).eval()
    nodes = get_model_onnx_nodes(model, x)
    assert nodes[4].op_type == 'TRTInstanceNormalization'
    assert nodes[4].domain == 'mmdeploy'


@pytest.mark.usefixtures('prepare_symbolics_ncnn')
class TestLinear:

    def check(self, nodes):
        print(nodes)
        exist = False
        for node in nodes:
            if node.op_type in ['Gemm', 'MatMul']:
                exist = True
                break

        assert exist is True

    def test_normal(self):
        x = torch.rand(1, 2, 3)
        w = torch.rand(2, 3)
        bias = torch.rand(2)
        model = OpModel(torch.nn.functional.linear, w, bias).eval()
        nodes = get_model_onnx_nodes(model, x)
        self.check(nodes)

    def test_no_bias(self):
        x = torch.rand(1, 2, 3)
        w = torch.rand(2, 3)
        model = OpModel(torch.nn.functional.linear, w).eval()
        nodes = get_model_onnx_nodes(model, x)
        self.check(nodes)


@pytest.mark.usefixtures('prepare_symbolics')
class TestSqueeze:

    def test_squeeze_default(self):
        x = torch.rand(1, 1, 2, 2)
        model = OpModel(torch.squeeze)
        nodes = get_model_onnx_nodes(model, x)
        assert nodes[0].attribute[0].ints == [0, 1]
        assert nodes[0].op_type == 'Squeeze'

    def test_squeeze(self):
        x = torch.rand(1, 1, 2, 2)
        model = OpModel(torch.squeeze, 0)
        nodes = get_model_onnx_nodes(model, x)
        assert nodes[0].attribute[0].ints == [0]
        assert nodes[0].op_type == 'Squeeze'


@pytest.mark.usefixtures('prepare_symbolics')
def test_hardsigmoid():
    x = torch.rand(1, 2, 3, 4)
    model = torch.nn.Hardsigmoid().eval()
    nodes = get_model_onnx_nodes(model, x)
    assert nodes[0].op_type == 'HardSigmoid'

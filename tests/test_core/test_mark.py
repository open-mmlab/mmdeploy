# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import onnx
import torch

from mmdeploy.core import RewriterContext, mark
from mmdeploy.core.optimizers import attribute_to_dict
from mmdeploy.utils.constants import IR, Backend

output_file = tempfile.NamedTemporaryFile(suffix='.onnx').name


def test_mark():

    @mark('add', inputs=['a', 'b'], outputs='c')
    def add(x, y):
        return torch.add(x, y)

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return add(x, y)

    model = TestModel().eval()

    # dummy input
    x = torch.rand(2, 3, 4)
    y = torch.rand(2, 3, 4)

    torch.onnx.export(model, (x, y), output_file)
    onnx_model = onnx.load(output_file)

    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'Mark'
    assert nodes[0].domain == 'mmdeploy'
    assert attribute_to_dict(nodes[0].attribute) == dict(
        dtype=1,
        func='add',
        func_id=0,
        id=0,
        type='input',
        name='a',
        shape=[2, 3, 4])

    assert nodes[1].op_type == 'Mark'
    assert nodes[1].domain == 'mmdeploy'
    assert attribute_to_dict(nodes[1].attribute) == dict(
        dtype=1,
        func='add',
        func_id=0,
        id=1,
        type='input',
        name='b',
        shape=[2, 3, 4])

    assert nodes[2].op_type == 'Add'

    assert nodes[3].op_type == 'Mark'
    assert nodes[3].domain == 'mmdeploy'
    assert attribute_to_dict(nodes[3].attribute) == dict(
        dtype=1,
        func='add',
        func_id=0,
        id=0,
        type='output',
        name='c',
        shape=[2, 3, 4])

    with RewriterContext(
            cfg=None, backend=Backend.TORCHSCRIPT.value,
            ir=IR.TORCHSCRIPT), torch.no_grad(), torch.jit.optimized_execution(
                True):
        torch.jit.trace(model, (x, y))

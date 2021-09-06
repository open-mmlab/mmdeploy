import os

import onnx
import pytest
import torch

from mmdeploy.core import mark
from mmdeploy.core.optimizers import attribute_to_dict

output_file = 'test_mark.onnx'


@pytest.fixture(autouse=True)
def clear_work_dir_after_test():
    # clear tmp output before test
    if os.path.exists(output_file):
        os.remove(output_file)
    yield
    # clear tmp output after test
    if os.path.exists(output_file):
        os.remove(output_file)


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
    assert nodes[0].domain == 'mmcv'
    assert attribute_to_dict(nodes[0].attribute) == dict(
        dtype=1,
        func='add',
        func_id=0,
        id=0,
        type='input',
        name='a',
        shape=[2, 3, 4])

    assert nodes[1].op_type == 'Mark'
    assert nodes[1].domain == 'mmcv'
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
    assert nodes[3].domain == 'mmcv'
    assert attribute_to_dict(nodes[3].attribute) == dict(
        dtype=1,
        func='add',
        func_id=0,
        id=0,
        type='output',
        name='c',
        shape=[2, 3, 4])


def test_extract():
    from mmdeploy.apis import extract_model

    @mark('add', outputs='z')
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

    extracted = extract_model(onnx_model, 'add:input', 'add:output')

    assert extracted.graph.input[0].name == 'x'
    assert extracted.graph.input[1].name == 'y'
    assert extracted.graph.output[0].name == 'z'
    assert extracted.graph.node[0].op_type == 'Add'

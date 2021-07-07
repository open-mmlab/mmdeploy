import os
import torch
import pytest

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
    from mmdeploy.utils import mark
    from mmdeploy.apis.utils import attribute_to_dict
    import onnx

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
        func='add', id=0, type='input', name='a')

    assert nodes[1].op_type == 'Mark'
    assert nodes[1].domain == 'mmcv'
    assert attribute_to_dict(nodes[1].attribute) == dict(
        func='add', id=1, type='input', name='b')

    assert nodes[2].op_type == 'Add'

    assert nodes[3].op_type == 'Mark'
    assert nodes[3].domain == 'mmcv'
    assert attribute_to_dict(nodes[3].attribute) == dict(
        func='add', id=0, type='output', name='c')


def test_extract():
    from mmdeploy.utils import mark
    from mmdeploy.apis import extract_model
    from mmdeploy.apis.utils import attribute_to_dict
    import onnx

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

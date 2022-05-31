# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import onnx
import torch

from mmdeploy.apis.onnx import extract_partition
from mmdeploy.core import mark

output_file = tempfile.NamedTemporaryFile(suffix='.onnx').name


def test_extract():

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

    extracted = extract_partition(onnx_model, 'add:input', 'add:output')

    assert extracted.graph.input[0].name == 'x'
    assert extracted.graph.input[1].name == 'y'
    assert extracted.graph.output[0].name == 'z'
    assert extracted.graph.node[0].op_type == 'Add'

# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from typing import Any, List, Tuple

import onnx
import pytest
import torch
import torch.nn as nn

from mmdeploy.apis.onnx.optimizer import \
    model_to_graph__custom_optimizer  # noqa
from mmdeploy.core import RewriterContext

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name

ort_cfg = dict(
    backend_config=dict(type='onnxruntime'), onnx_config=dict(type='onnx'))


def _find_next_node(start: int, nodes: List, op_type: str) -> Tuple[Any, int]:
    for idx, n in enumerate(nodes[start:]):
        if n.op_type == op_type:
            return n, idx
    return None, -1


def test_merge_shape_concate():
    pytest.importorskip('mmdeploy.backend.torchscript.ts_optimizer.onnx')

    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        opt_pass = ts_optimizer.onnx._jit_pass_merge_shape_concate
    except ImportError:
        pytest.skip('pass not found.')

    def _optimize_onnx(ctx, graph, params_dict, torch_out):
        opt_pass(graph)
        return graph, params_dict, torch_out

    class TestModel(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.new_zeros(x.shape[-2:])

    model = TestModel()
    x = torch.rand(1, 3, 4, 8)

    with RewriterContext({}, onnx_custom_passes=_optimize_onnx):
        torch.onnx.export(
            model,
            x,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dict(input={
                2: 'h',
                3: 'w'
            }),
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    nodes = graph.node
    shape_idx = 0
    for n in nodes:
        if n.op_type != 'Shape':
            shape_idx += 1
        else:
            break

    assert shape_idx < len(nodes)
    assert nodes[shape_idx + 1].op_type == 'Gather'
    assert nodes[shape_idx + 2].op_type == 'ConstantOfShape'


def test_peephole():
    pytest.importorskip('mmdeploy.backend.torchscript.ts_optimizer.onnx')

    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        opt_pass = ts_optimizer.onnx._jit_pass_onnx_peephole
    except ImportError:
        pytest.skip('pass not found.')

    def _optimize_onnx(ctx, graph, params_dict, torch_out):
        opt_pass(graph)
        return graph, params_dict, torch_out

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):

            x = x.int()
            x = x.int()
            x = x.float()

            x = x.view(10, -1)
            y = x.view(2, -1)
            z = x.view(3, -1)

            return y, z

    model = TestModel()
    x = torch.rand(2, 3, 5)

    with RewriterContext({}, onnx_custom_passes=_optimize_onnx):
        torch.onnx.export(
            model,
            x,
            onnx_file,
            input_names=['input'],
            output_names=['output1', 'output2'],
            dynamic_axes=dict(input={
                0: 'b',
                1: 'c',
                2: 'w'
            }),
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    nodes = graph.node

    node, idx = _find_next_node(0, nodes, 'Cast')
    assert node is not None
    assert node.attribute[0].i == 6

    node, idx = _find_next_node(idx + 1, nodes, 'Cast')
    assert node is not None
    assert node.attribute[0].i == 1

    node, idx = _find_next_node(idx + 1, nodes, 'Reshape')
    assert node is not None

    node, idx = _find_next_node(idx + 1, nodes, 'Reshape')
    assert node is not None


def test_flatten_cls_head():
    pytest.importorskip('mmdeploy.backend.torchscript.ts_optimizer.onnx')

    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        opt_pass = ts_optimizer.onnx._jit_pass_flatten_cls_head
    except ImportError:
        pytest.skip('pass not found.')

    def _optimize_onnx(ctx, graph, params_dict, torch_out):
        opt_pass(graph)
        return graph, params_dict, torch_out

    class TestModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            batch = x.size(0)
            gap = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            gap = gap.reshape(batch, -1)
            return gap + 0  # gap should not be the output

    model = TestModel()
    x = torch.rand(1, 4, 8, 8)

    with RewriterContext(ort_cfg, onnx_custom_passes=_optimize_onnx):
        torch.onnx.export(
            model,
            x,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dict(input={
                2: 'h',
                3: 'w'
            }),
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    nodes = graph.node

    node, idx = _find_next_node(0, nodes, 'GlobalAveragePool')
    assert node is not None

    node, idx = _find_next_node(idx + 1, nodes, 'Flatten')
    assert node is not None


def test_fuse_select_assign():
    pytest.importorskip('mmdeploy.backend.torchscript.ts_optimizer.onnx')

    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        opt_pass = ts_optimizer.onnx._jit_pass_fuse_select_assign
    except ImportError:
        pytest.skip('pass not found.')

    def _optimize_onnx(ctx, graph, params_dict, torch_out):
        opt_pass(graph, params_dict)
        return graph, params_dict, torch_out

    class TestModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            z = x / 2
            y = torch.zeros_like(x)
            y[x < 0.5] = z[x < 0.5]
            return y

    model = TestModel()
    x = torch.rand(1, 4, 8, 8)

    with RewriterContext({}, onnx_custom_passes=_optimize_onnx):
        torch.onnx.export(
            model,
            x,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dict(input={
                2: 'h',
                3: 'w'
            }),
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    nodes = graph.node

    node, _ = _find_next_node(0, nodes, 'Where')
    assert node is not None


def test_common_subgraph_elimination():
    pytest.importorskip('mmdeploy.backend.torchscript.ts_optimizer.onnx')

    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        opt_pass = ts_optimizer.onnx._jit_pass_common_subgraph_elimination
    except ImportError:
        pytest.skip('pass not found.')

    def _optimize_onnx(ctx, graph, params_dict, torch_out):
        opt_pass(graph, params_dict)
        return graph, params_dict, torch_out

    class TestModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            y = x.unsqueeze(1)
            z = x.unsqueeze(1)
            return y + z

    model = TestModel()
    x = torch.rand(1, 2, 3)

    with RewriterContext({}, onnx_custom_passes=_optimize_onnx):
        torch.onnx.export(
            model,
            x,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dict(input={
                1: 'h',
                2: 'w'
            }),
            opset_version=11)

    onnx_model = onnx.load(onnx_file)
    graph = onnx_model.graph
    nodes = graph.node

    unsqueeze_count = 0
    for n in nodes:
        if n.op_type == 'Unsqueeze':
            unsqueeze_count += 1
    assert unsqueeze_count == 1

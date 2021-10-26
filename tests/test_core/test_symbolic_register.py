import tempfile

import onnx
import pytest
import torch
from torch.autograd import Function

import mmdeploy
from mmdeploy.core import SYMBOLIC_REGISTER, RewriterContext
from mmdeploy.core.rewriters.symbolic_register import SymbolicRegister

output_file = tempfile.NamedTemporaryFile(suffix='.onnx').name


@pytest.fixture(autouse=True, scope='module')
def create_custom_module():

    class TestFunc(Function):

        @staticmethod
        def symbolic(g, x, val):
            return g.op('mmcv::symbolic_old', x, val_i=val)

        @staticmethod
        def forward(ctx, x, val):
            return x + val

    # put TestFunc in an module so we can found it
    # could be any module
    mmdeploy.TestFunc = TestFunc

    yield

    del mmdeploy.TestFunc


def test_symbolic_register():
    test_func = mmdeploy.TestFunc.apply

    @SYMBOLIC_REGISTER.register_symbolic('mmdeploy.TestFunc', backend='ncnn')
    @SYMBOLIC_REGISTER.register_symbolic('mmdeploy.TestFunc')
    def symbolic_testfunc_default(symbolic_wrapper, g, x, val):
        assert hasattr(symbolic_wrapper, 'cfg')
        return g.op('mmcv::symbolic_testfunc_default', x, val_i=val)

    @SYMBOLIC_REGISTER.register_symbolic(
        'mmdeploy.TestFunc', backend='tensorrt')
    def symbolic_testfunc_tensorrt(symbolic_wrapper, g, x, val):
        return g.op('mmcv::symbolic_testfunc_tensorrt', x, val_i=val)

    @SYMBOLIC_REGISTER.register_symbolic(
        'cummax', is_pytorch=True, arg_descriptors=['v', 'i'])
    def symbolic_cummax(symbolic_wrapper, g, input, dim):
        return g.op('mmcv::cummax_default', input, dim_i=dim, outputs=2)

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cummax(test_func(x, 5), dim=1)

    model = TestModel().eval()

    # dummy input
    x = torch.rand(2, 3, 4)

    # default
    cfg = dict()
    with RewriterContext(cfg=cfg, opset=11):
        torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'symbolic_testfunc_default'
    assert nodes[0].domain == 'mmcv'
    assert nodes[1].op_type == 'cummax_default'
    assert nodes[1].domain == 'mmcv'

    # ncnn
    with RewriterContext(cfg=cfg, backend='ncnn', opset=11):
        torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'symbolic_testfunc_default'
    assert nodes[0].domain == 'mmcv'
    assert nodes[1].op_type == 'cummax_default'
    assert nodes[1].domain == 'mmcv'

    # tensorrt
    with RewriterContext(cfg=cfg, backend='tensorrt', opset=11):
        torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'symbolic_testfunc_tensorrt'
    assert nodes[0].domain == 'mmcv'
    assert nodes[1].op_type == 'cummax_default'
    assert nodes[1].domain == 'mmcv'


def test_unregister():
    test_func = mmdeploy.TestFunc.apply

    @SYMBOLIC_REGISTER.register_symbolic('mmdeploy.TestFunc')
    def symbolic_testfunc_default(symbolic_wrapper, g, x, val):
        return g.op('mmcv::symbolic_testfunc_default', x, val_i=val)

    @SYMBOLIC_REGISTER.register_symbolic(
        'cummax', is_pytorch=True, arg_descriptors=['v', 'i'])
    def symbolic_cummax(symbolic_wrapper, g, input, dim):
        return g.op('mmcv::cummax_default', input, dim_i=dim, outputs=2)

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cummax(x, dim=1)

    class TestModel2(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return test_func(x, 5)

    model = TestModel().eval()
    x = torch.rand(2, 3, 4)

    with RewriterContext(cfg={}, opset=11):
        torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'cummax_default'
    assert nodes[0].domain == 'mmcv'

    with pytest.raises(RuntimeError):
        torch.onnx.export(model, x, output_file, opset_version=11)

    model = TestModel2().eval()
    with RewriterContext(cfg={}, opset=11):
        torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'symbolic_testfunc_default'
    assert nodes[0].domain == 'mmcv'

    torch.onnx.export(model, x, output_file, opset_version=11)
    onnx_model = onnx.load(output_file)
    nodes = onnx_model.graph.node
    assert nodes[0].op_type == 'symbolic_old'
    assert nodes[0].domain == 'mmcv'


def test_register_empty_symbolic():
    symbolic_register = SymbolicRegister()

    @symbolic_register.register_symbolic('mmdeploy.EmptyFunction')
    def symbolic_testfunc_default(symbolic_wrapper, g, x, val):
        return g.op('mmcv::symbolic_testfunc_default', x, val_i=val)

    symbolic_register.enter()
    assert len(symbolic_register._extra_symbolic) == 0
    symbolic_register.exit()

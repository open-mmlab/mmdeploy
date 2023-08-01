# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mmengine import Config
from packaging.version import parse

from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapFunction, backend_checker,
                                 get_rewrite_outputs)

deploy_cfg_ncnn = Config(
    dict(
        onnx_config=dict(input_shape=None),
        backend_config=dict(type='ncnn', model_inputs=None, use_vulkan=False),
        codebase_config=dict(type='mmdet', task='ObjectDetection')))


def get_trt_config(output_names, shape, dynamic_axes=None):
    deploy_cfg_tensorrt = Config(
        dict(
            onnx_config=dict(
                input_shape=None,
                output_names=output_names,
                dynamic_axes=dynamic_axes),
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    fp16_mode=False, max_workspace_size=1 << 20),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            input=dict(
                                min_shape=shape,
                                opt_shape=shape,
                                max_shape=shape)))
                ]),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    return deploy_cfg_tensorrt


@backend_checker(Backend.NCNN)
def test_get_attribute():

    def model_func(tensor):
        x = tensor.size()
        assert isinstance(x[0], int) and not isinstance(x[0], torch.Tensor)
        return torch.tensor(x)

    input = torch.zeros([1, 2, 3, 4])
    wrapped_func = WrapFunction(model_func)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'tensor': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '
    'outputs: {}'.format(rewrite_outputs)


@backend_checker(Backend.NCNN)
def test_group_norm_ncnn():
    input = torch.rand([1, 2, 2, 2])
    weight = torch.rand([2])
    bias = torch.rand([2])
    model_output = F.group_norm(input, 1, weight, bias, 1e-05)

    def group_norm_caller(input):
        return F.group_norm(input, 1, weight, bias)

    wrapped_func = WrapFunction(group_norm_caller)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert np.allclose(model_output, rewrite_output[0], rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_chunk_ncnn():
    input = torch.rand(1, 16, 16, 16)

    model_output = input.chunk(2, dim=1)

    def chunk_caller(input):
        return input.chunk(2, dim=1)

    wrapped_func = WrapFunction(chunk_caller)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert len(model_output) == len(rewrite_output)
    for i in range(len(model_output)):
        assert np.allclose(
            model_output[i], rewrite_output[i], rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_interpolate_static():
    input = torch.rand([1, 2, 2, 2])
    model_output = F.interpolate(input, scale_factor=[2, 2])

    def interpolate_caller(*arg, **kwargs):
        return F.interpolate(*arg, **kwargs)

    wrapped_func = WrapFunction(interpolate_caller, size=[4, 4])
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert np.allclose(model_output, rewrite_output[0], rtol=1e-03, atol=1e-05)


@backend_checker(Backend.RKNN)
def test_interpolate__rknn():
    input = torch.rand([1, 2, 2, 2])
    model_output = F.interpolate(input, scale_factor=[2, 2])

    def interpolate_caller(*arg, **kwargs):
        return F.interpolate(*arg, **kwargs)

    deploy_cfg = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='rknn', model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    wrapped_func = WrapFunction(interpolate_caller, size=[4, 4])
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    assert np.allclose(model_output, rewrite_output[0], rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_linear_ncnn():
    input = torch.rand([1, 2, 2])
    weight = torch.rand([2, 2])
    bias = torch.rand([2])
    model_output = F.linear(input, weight=weight, bias=bias)

    def linear_caller(*arg, **kwargs):
        return F.linear(*arg, **kwargs)

    wrapped_func = WrapFunction(linear_caller, weight=weight, bias=bias)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert np.allclose(model_output, rewrite_output[0], rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_norm_ncnn():
    import onnx

    import mmdeploy.apis.ncnn as ncnn_apis
    from mmdeploy.utils.test import get_onnx_model

    input = torch.rand(1, 17, 24)
    wrapped_func = WrapFunction(torch.norm, p='fro', dim=2, keepdim=True)
    model_inputs = {'input': input}
    ir_file_path = get_onnx_model(wrapped_func, model_inputs, deploy_cfg_ncnn)
    assert osp.exists(ir_file_path)
    onnx_model = onnx.load(ir_file_path)
    nodes = onnx_model.graph.node
    assert nodes[-1].name.startswith('ReduceL2')
    ncnn_files_prefix = osp.splitext(ir_file_path)[0]
    ncnn_apis.from_onnx(ir_file_path, ncnn_files_prefix)
    param_path, bin_path = ncnn_apis.get_output_model_file(ir_file_path)
    assert osp.exists(param_path)
    assert osp.exists(bin_path)


@backend_checker(Backend.TENSORRT)
def test_repeat_static():
    input = torch.rand([1])

    def model_func(input):
        return torch.Tensor.repeat(input, 4)

    wrapped_func = WrapFunction(model_func)

    model_output = model_func(input)

    deploy_cfg = get_trt_config(['output'], [1])

    rewrite_output, is_backend_output = get_rewrite_outputs(
        wrapped_func, model_inputs={'input': input}, deploy_cfg=deploy_cfg)

    if is_backend_output:
        rewrite_output = rewrite_output[0].detach().cpu()

        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_output is not None


@backend_checker(Backend.NCNN)
def test_size_of_tensor_static():

    def model_func(input):
        x = torch.Tensor.size(input)
        assert isinstance(x[0], int) and not isinstance(x[0], torch.Tensor)
        return torch.tensor(x)

    input = torch.zeros([1, 2, 3, 4])
    wrapped_func = WrapFunction(model_func)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn,
        run_with_backend=True)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '
    'outputs: {}'.format(rewrite_outputs)


@backend_checker(Backend.ASCEND)
def test_size__ascend():

    def model_func(input):
        x = torch.Tensor.size(input, -1)
        return torch.tensor(x)

    input = torch.zeros([1, 2, 3, 4])
    deploy_cfg_ascend = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(
                type='ascend',
                model_inputs=[dict(input_shapes=dict(input=input.shape))]),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    wrapped_func = WrapFunction(model_func)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ascend,
        run_with_backend=True)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '
    'outputs: {}'.format(rewrite_outputs)


class TestTopk:

    input = torch.rand(1, 5, 5, 5)

    @backend_checker(Backend.NCNN)
    @pytest.mark.parametrize('k', [1, 3, 4])
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_topk_ncnn(self, dim, k):

        model_output = torch.Tensor.topk(TestTopk.input, k, dim).values

        def model_func(input):
            x = input.topk(k, dim)
            return x.indices, x.values

        wrapped_func = WrapFunction(model_func)

        # mmdeploy.pytorch.functions.topk.topk_dynamic
        output, _ = get_rewrite_outputs(
            wrapped_func,
            model_inputs={'input': TestTopk.input},
            deploy_cfg=deploy_cfg_ncnn,
            run_with_backend=True)
        assert np.allclose(model_output, output[0], rtol=1e-03, atol=1e-05)

    @backend_checker(Backend.TENSORRT)
    @pytest.mark.parametrize('k', [1, 3, 4])
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_topk_tensorrt(self, dim, k):
        model_output = torch.Tensor.topk(TestTopk.input, k, dim).values

        def model_func(input):
            x = input.topk(k, dim)
            return x.indices, x.values

        wrapped_func = WrapFunction(model_func)

        # mmdeploy.pytorch.functions.topk.topk_static
        deploy_cfg_tensorrt = get_trt_config(['indices', 'values'],
                                             [1, 5, 5, 5])
        output, is_backend_output = get_rewrite_outputs(
            wrapped_func,
            model_inputs={'input': TestTopk.input},
            deploy_cfg=deploy_cfg_tensorrt)

        if is_backend_output:
            output = output[1].detach().cpu()

            assert np.allclose(model_output, output, rtol=1e-03, atol=1e-05)
        else:
            assert output is not None


@backend_checker(Backend.TENSORRT)
@pytest.mark.parametrize('shape', [[2, 2], [4, 2], [2, 4], [2, 4, 2]])
@pytest.mark.parametrize('diagonal', [0, 1, -1])
def test_triu_trt(shape, diagonal):

    input = torch.rand(shape)
    model_output = torch.triu(input=input, diagonal=diagonal)

    def triu_caller(*arg, **kwargs):
        return torch.triu(*arg, **kwargs)

    wrapped_func = WrapFunction(triu_caller, diagonal=diagonal)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=get_trt_config(['output'], shape=shape),
        run_with_backend=True)
    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0].detach().cpu()

        assert np.allclose(
            model_output, rewrite_outputs, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None


@backend_checker(Backend.NCNN)
@pytest.mark.parametrize(
    'input',
    [torch.rand(1, 16, 16), torch.rand(1, 3, 16, 16)])
@pytest.mark.parametrize('dim', [1, 2])
def test_normalize_ncnn(input, dim):
    import mmdeploy.apis.ncnn as ncnn_apis
    from mmdeploy.utils.test import get_onnx_model

    def norm_func(input, dim):
        return F.normalize(input, p=2, dim=dim)

    wrapped_func = WrapFunction(norm_func, dim=dim)
    model_inputs = {'input': input}
    ir_file_path = get_onnx_model(wrapped_func, model_inputs, deploy_cfg_ncnn)
    assert osp.exists(ir_file_path)
    ncnn_files_prefix = osp.splitext(ir_file_path)[0]
    ncnn_apis.from_onnx(ir_file_path, ncnn_files_prefix)
    param_path, bin_path = ncnn_apis.get_output_model_file(ir_file_path)
    assert osp.exists(param_path)
    assert osp.exists(bin_path)


@backend_checker(Backend.ASCEND)
def test_getitem__ascend():

    input = torch.rand(1, 2, 3)

    def tensor_getitem(x):
        return x[..., -1]

    # create wrapped model
    wrapped_func = WrapFunction(tensor_getitem)
    import tempfile

    import onnx

    from mmdeploy.core import RewriterContext
    onnx_file = tempfile.NamedTemporaryFile(suffix='onnx').name

    # convert model
    with RewriterContext(
            cfg={}, backend=Backend.ASCEND.value, opset=11), torch.no_grad():
        torch.onnx.export(wrapped_func, input, onnx_file, opset_version=11)
    onnx_model = onnx.load(onnx_file)
    nodes = onnx_model.graph.node
    assert nodes is not None


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize(
    'input',
    [torch.rand(1, 16, 16), torch.rand(1, 3, 16, 16)])
def test_masked_fill_onnxruntime(input):
    mask = input > 0
    value = float('-inf')

    def masked_fill_caller(*arg, **kwargs):
        return torch.masked_fill(*arg, **kwargs)

    deploy_cfg_ort = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    wrapped_func = WrapFunction(masked_fill_caller, mask=mask, value=value)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ort,
        run_with_backend=True)
    assert rewrite_output is not None


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.skipif(
    parse(torch.__version__) < parse('1.9.0'), reason='requires torch>1.8.0')
@pytest.mark.parametrize('x', [torch.rand(1, 3, 16, 16)])
@pytest.mark.parametrize('y', [torch.rand(1, 3, 4, 4)])
def test_tensor_setitem(x, y):
    import onnx

    from mmdeploy.utils.test import get_onnx_model

    def setitem_slice(x, y):
        H, W = y.shape[2:]
        x[:, :, 2:H + 2, 2:W + 2] = y
        return x

    wrapped_func = WrapFunction(setitem_slice)
    model_inputs = {'x': x, 'y': y}

    deploy_cfg = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    ir_file_path = get_onnx_model(wrapped_func, model_inputs, deploy_cfg)

    onnx_model = onnx.load(ir_file_path)
    nodes = onnx_model.graph.node
    for node in nodes:
        assert node.op_type != 'ScatterND'


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.skipif(
    parse(torch.__version__) < parse('1.9.0'), reason='requires torch>1.8.0')
@pytest.mark.parametrize('x', [torch.rand(1, 3, 16, 16)])
def test_tensor_setitem_scalar(x):
    import onnx

    from mmdeploy.utils.test import get_onnx_model

    def setitem_slice(x):
        H, W = x.shape[-2:]
        x[:, 1:3] = 1
        x[:, :, 4:H - 4, 4:W - 4] = x.new_tensor(2)
        return x

    wrapped_func = WrapFunction(setitem_slice)
    model_inputs = {'x': x}

    deploy_cfg = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    ir_file_path = get_onnx_model(wrapped_func, model_inputs, deploy_cfg)

    onnx_model = onnx.load(ir_file_path)
    nodes = onnx_model.graph.node
    for node in nodes:
        assert node.op_type != 'ScatterND'


@pytest.mark.parametrize('output_size', [1, 3])
def test_adaptive_avg_pool2d(output_size):
    input = torch.rand(1, 3, 6, 6)
    model = WrapFunction(F.adaptive_avg_pool2d, output_size=output_size)
    pytorch_output = model(input)
    deploy_cfg_ort = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ort,
        run_with_backend=True)
    assert torch.allclose(pytorch_output, rewrite_output[0])


@backend_checker(Backend.TENSORRT)
def test_scaled_dot_product_attention():
    L = 10
    B = 1
    E = 4
    q = k = v = torch.rand(B, L, E)
    attn_mask = torch.rand(B, L, L)

    from torch.nn.functional import _scaled_dot_product_attention
    model = WrapFunction(_scaled_dot_product_attention)
    pytorch_output = model(q, k, v, attn_mask)
    deploy_cfg_ort = Config(
        dict(
            onnx_config=dict(
                input_shape=None,
                input_names=['q', 'k', 'v', 'attn_mask'],
                output_names=['output', 'attn']),
            backend_config=dict(
                type='tensorrt',
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            q=dict(
                                min_shape=q.shape,
                                opt_shape=q.shape,
                                max_shape=q.shape),
                            k=dict(
                                min_shape=k.shape,
                                opt_shape=k.shape,
                                max_shape=k.shape),
                            v=dict(
                                min_shape=v.shape,
                                opt_shape=v.shape,
                                max_shape=v.shape),
                            attn_mask=dict(
                                min_shape=attn_mask.shape,
                                opt_shape=attn_mask.shape,
                                max_shape=attn_mask.shape)))
                ]),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={
            'q': q,
            'k': k,
            'v': v,
            'attn_mask': attn_mask
        },
        deploy_cfg=deploy_cfg_ort,
        run_with_backend=True)
    assert torch.allclose(pytorch_output[0],
                          rewrite_output[0].to(pytorch_output[0].device))


@backend_checker(Backend.TENSORRT)
@pytest.mark.parametrize('num', [5, 7])
def test_mod__tensorrt(num):
    input = torch.rand(1, 3, 6, 6).cuda()
    model = WrapFunction(lambda input: input % num)
    pytorch_output = model(input)
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'input': input},
        deploy_cfg=get_trt_config(['output'], shape=[1, 3, 6, 6]),
        run_with_backend=True)
    assert torch.allclose(
        pytorch_output, rewrite_output[0], rtol=1e-3, atol=1e-5)


@backend_checker(Backend.TENSORRT)
def test_prepare_onnx_paddings__tensorrt():
    input = torch.rand(1, 3, 6, 6).cuda()

    def _pad_(x):
        a, b = [torch.tensor(2)] * 2
        x = torch.nn.ZeroPad2d((0, a, 0, b))(x)
        return x

    model = WrapFunction(_pad_)
    pytorch_output = model(input)
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'x': input},
        deploy_cfg=get_trt_config(['output'], shape=[1, 3, 6, 6]),
        run_with_backend=True)
    assert torch.allclose(
        pytorch_output, rewrite_output[0], rtol=1e-3, atol=1e-5)


@backend_checker(Backend.ONNXRUNTIME)
@pytest.mark.parametrize('dim', [0, -1])
@pytest.mark.parametrize('keepdim', [True, False])
def test_any__default(dim, keepdim):
    input = torch.rand(2, 4)
    model = WrapFunction(lambda input: input.any(dim, keepdim=keepdim))
    pytorch_output = model(input)
    deploy_cfg_ort = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ort,
        run_with_backend=True)
    assert pytorch_output.dtype == rewrite_output[0].dtype
    assert torch.allclose(
        pytorch_output.float(),
        rewrite_output[0].float(),
        rtol=1e-3,
        atol=1e-5)


@backend_checker(Backend.ONNXRUNTIME)
def test_linspace__default():
    import random

    deploy_cfg_ort = Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type='onnxruntime')))

    def linspace_caller(*arg, **kwargs):
        return torch.linspace(*arg, **kwargs)

    steps_list = [1, random.randint(1, 1000)]
    for steps in steps_list:
        start = random.random() * 100
        end = random.random() * 100 + start

        model_output = linspace_caller(start=start, end=end, steps=steps)

        wrapped_func = WrapFunction(
            linspace_caller, start=start, end=end, steps=steps)

        rewrite_outputs, is_backend_output = get_rewrite_outputs(
            wrapped_func,
            model_inputs={},
            deploy_cfg=deploy_cfg_ort,
            run_with_backend=True)

        if is_backend_output:
            rewrite_outputs = rewrite_outputs[0]

        assert np.allclose(
            model_output, rewrite_outputs, rtol=1e-03, atol=1e-05)


@backend_checker(Backend.TENSORRT)
@pytest.mark.parametrize('dtype', [torch.bool, torch.float32])
@pytest.mark.parametrize('dynamic_axes',
                         [None, dict(input=dict({
                             0: 'dim0',
                             1: 'dim1'
                         }))])
def test_cat__tensorrt(dtype, dynamic_axes):
    input = torch.rand(2, 4)
    model = WrapFunction(lambda input: torch.cat(
        [input.to(dtype), input.to(dtype)], -1))
    pytorch_output = model(input)
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'input': input},
        deploy_cfg=get_trt_config(['output'],
                                  shape=[2, 4],
                                  dynamic_axes=dynamic_axes),
        run_with_backend=True)
    assert pytorch_output.dtype == rewrite_output[0].dtype
    assert torch.allclose(
        pytorch_output.cpu().float(),
        rewrite_output[0].cpu().float(),
        rtol=1e-3,
        atol=1e-5)


@backend_checker(Backend.TENSORRT)
def test_copy__default():
    import copy
    input = torch.rand(2, 4)
    model = WrapFunction(
        lambda input: [copy.deepcopy(input) for i in range(3)])
    pytorch_output = model(input)
    rewrite_output, _ = get_rewrite_outputs(
        model,
        model_inputs={'input': input},
        deploy_cfg=get_trt_config(['output'], shape=[2, 4], dynamic_axes=None),
        run_with_backend=True)
    for pytorch_out, rewrite_out in zip(pytorch_output, rewrite_output):
        assert torch.allclose(
            pytorch_out.cpu().float(),
            rewrite_out.cpu().float(),
            rtol=1e-3,
            atol=1e-5)

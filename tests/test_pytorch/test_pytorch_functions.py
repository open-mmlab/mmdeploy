# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapFunction, backend_checker,
                                 get_rewrite_outputs)

deploy_cfg_ncnn = mmcv.Config(
    dict(
        onnx_config=dict(input_shape=None),
        backend_config=dict(type='ncnn', model_inputs=None, use_vulkan=False),
        codebase_config=dict(type='mmdet', task='ObjectDetection')))


def get_trt_config(output_names, shape):
    deploy_cfg_tensorrt = mmcv.Config(
        dict(
            onnx_config=dict(input_shape=None, output_names=output_names),
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
def test_triu_trt(shape):

    input = torch.rand(shape)

    def triu_caller(*arg, **kwargs):
        return torch.triu(*arg, **kwargs)

    wrapped_func = WrapFunction(triu_caller, diagonal=1)
    import tempfile

    import onnx

    from mmdeploy.core import RewriterContext
    onnx_file = tempfile.NamedTemporaryFile(suffix='onnx').name
    with RewriterContext(
            cfg=get_trt_config('output', shape),
            backend=Backend.TENSORRT.value,
            opset=11), torch.no_grad():
        torch.onnx.export(wrapped_func, input, onnx_file, opset_version=11)
    onnx_model = onnx.load(onnx_file)
    nodes = onnx_model.graph.node
    assert nodes is not None


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

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest
import torch
import torch.nn.functional as func

from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapFunction, backend_checker,
                                 get_rewrite_outputs)

deploy_cfg_ncnn = mmcv.Config(
    dict(
        onnx_config=dict(input_shape=None),
        backend_config=dict(type='ncnn', model_inputs=None),
        codebase_config=dict(type='mmdet', task='ObjectDetection')))


def get_trt_config(output_names, shape):
    import tensorrt
    deploy_cfg_tensorrt = mmcv.Config(
        dict(
            onnx_config=dict(input_shape=None, output_names=output_names),
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    fp16_mode=False,
                    log_level=tensorrt.Logger.INFO,
                    max_workspace_size=1 << 20),
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
        return x[0] * tensor

    input = torch.zeros([1, 2, 3, 4])
    wrapped_func = WrapFunction(model_func)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'tensor': input},
        deploy_cfg=deploy_cfg_ncnn)

    assert rewrite_outputs is not None, 'Got unexpected rewrite '
    'outputs: {}'.format(rewrite_outputs)


@backend_checker(Backend.NCNN)
def test_group_norm_ncnn():
    input = torch.rand([1, 2, 2, 2])
    weight = torch.rand([2])
    bias = torch.rand([2])
    model_output = func.group_norm(input, 1, weight, bias, 1e-05)

    def group_norm_caller(input):
        return func.group_norm(input, 1, weight, bias)

    wrapped_func = WrapFunction(group_norm_caller)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn)

    assert np.allclose(model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_interpolate_static():
    input = torch.rand([1, 2, 2, 2])
    model_output = func.interpolate(input, scale_factor=[2, 2])

    def interpolate_caller(*arg, **kwargs):
        return func.interpolate(*arg, **kwargs)

    wrapped_func = WrapFunction(interpolate_caller, size=[4, 4])
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn)

    assert np.allclose(model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@backend_checker(Backend.NCNN)
def test_linear_ncnn():
    input = torch.rand([1, 2, 2])
    weight = torch.rand([2, 2])
    bias = torch.rand([2])
    model_output = func.linear(input, weight=weight, bias=bias)

    def linear_caller(*arg, **kwargs):
        return func.linear(*arg, **kwargs)

    wrapped_func = WrapFunction(linear_caller, weight=weight, bias=bias)
    rewrite_output, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn)

    assert np.allclose(model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@backend_checker(Backend.TENSORRT)
def test_repeat_static():
    input = torch.rand([1])

    def model_func(input):
        return torch.Tensor.repeat(input, 4)

    wrapped_func = WrapFunction(model_func)

    model_output = model_func(input)

    deploy_cfg = get_trt_config(['output'], [1])

    rewrite_output, is_backend_ouptut = get_rewrite_outputs(
        wrapped_func, model_inputs={'input': input}, deploy_cfg=deploy_cfg)

    if is_backend_ouptut:
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
        return x[0] * input

    input = torch.zeros([1, 2, 3, 4])
    wrapped_func = WrapFunction(model_func)
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_func,
        model_inputs={'input': input},
        deploy_cfg=deploy_cfg_ncnn)

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
            deploy_cfg=deploy_cfg_ncnn)

        assert np.allclose(model_output, output[1], rtol=1e-03, atol=1e-05)

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

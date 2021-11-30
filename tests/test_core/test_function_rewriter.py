# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER, RewriterContext
from mmdeploy.core.rewriters.function_rewriter import FunctionRewriter


def test_function_rewriter():

    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([2, 4, 6, 8, 10])

    @FUNCTION_REWRITER.register_rewriter(
        func_name='torch.mul', backend='tensorrt')
    @FUNCTION_REWRITER.register_rewriter(
        func_name='torch.add', backend='tensorrt')
    def sub_func(rewriter, x, y):
        assert hasattr(rewriter, 'cfg')
        assert hasattr(rewriter, 'origin_func')
        return x - y

    cfg = dict()
    with RewriterContext(cfg, backend='tensorrt'):
        result = torch.add(x, y)
        # replace add with sub
        torch.testing.assert_allclose(result, x - y)
        result = torch.mul(x, y)
        # replace add with sub
        torch.testing.assert_allclose(result, x - y)

    result = torch.add(x, y)
    # recovery origin function
    torch.testing.assert_allclose(result, x + y)

    with RewriterContext(cfg):
        result = torch.add(x, y)
        # replace should not happen with wrong backend
        torch.testing.assert_allclose(result, x + y)

    # test different config
    @FUNCTION_REWRITER.register_rewriter(
        func_name='torch.Tensor.add', backend='default')
    def mul_func_class(rewriter, x, y):
        return x * y

    with RewriterContext(cfg, backend='tensorrt'):
        result = x.add(y)
        # replace add with multi
        torch.testing.assert_allclose(result, x * y)

    result = x.add(y)
    # recovery origin function
    torch.testing.assert_allclose(result, x + y)

    with RewriterContext(cfg):
        result = x.add(y)
        # replace add with multi
        torch.testing.assert_allclose(result, x * y)

    # test origin_func
    @FUNCTION_REWRITER.register_rewriter(
        func_name='torch.add', backend='default')
    def origin_add_func(rewriter, x, y, **kwargs):
        return rewriter.origin_func(x, y, **kwargs) + 1

    with RewriterContext(cfg):
        result = torch.add(x, y)
        # replace with origin + 1
        torch.testing.assert_allclose(result, x + y + 1)

    # remove torch.add
    del FUNCTION_REWRITER._origin_functions[-1]
    torch.testing.assert_allclose(torch.add(x, y), x + y)


def test_rewrite_empty_function():
    function_rewriter = FunctionRewriter()

    @function_rewriter.register_rewriter(func_name='torch.abcdefghijklmn')
    def func(rewriter, x, y):
        return x + y

    function_rewriter.enter()
    assert len(function_rewriter._origin_functions) == 0
    function_rewriter.exit()

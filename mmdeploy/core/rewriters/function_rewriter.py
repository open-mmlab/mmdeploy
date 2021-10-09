import logging
from typing import Callable, Dict

from mmcv.utils import Registry

from .register_utils import eval_with_import


# caller wrapper
class FuncCaller(object):
    """The function wrapper used to call rewrite function.

    Args:
        cfg (Dict): Config dictionary of deployment.
    """
    func_name = None
    backend = None
    func = None

    def __init__(self, cfg: Dict, **kwargs):
        self.cfg = cfg
        try:
            origin_func = eval_with_import(self.func_name)
        except Exception:
            origin_func = None
            logging.warning(
                f'Can not find {self.func_name}, function rewrite will '
                'not be applied')
        self.origin_func = origin_func
        [setattr(self, k, v) for k, v in kwargs.items()]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# builder of register
def build_caller(func_name: str, backend: str, cfg: Dict, registry: Registry,
                 **kwargs) -> FuncCaller:
    """Build the caller of the given function name and backend.

    Args:
        func_name (str): The function name/path to rewrite.
        backend (str): The inference engine name.
        cfg (Dict): Config dictionary of deployment.
        registry (Registry): The registry to apply this build function.

    Returns:
        FuncCaller: The caller instance of rewrite.
    """
    func_caller = registry.module_dict[func_name + '@' + backend]
    assert func_caller is not None, f'{func_name} with {backend} not exist.'
    return func_caller(cfg, **kwargs)


# create register
FUNCTION_REWRITER = Registry('func_rewriters', build_func=build_caller)


# caller decorator
def register_rewriter(func_name: str,
                      backend: str = 'default',
                      **kwargs) -> Callable:
    """Decorator of the rewrite function.

    Args:
        func_name (str): The function name/path to rewrite.
        backend (str): The inference engine name.

    Returns:
        Callable: The process of registering function.

    Examples:
        >>> @FUNCTION_REWRITER.register_rewriter(
        >>>     func_name='torch.Tensor.size', backend='ncnn')
        >>> def size_of_tensor_static(ctx, self, *args):
        >>>     ret = ctx.origin_func(self, *args)
        >>>     if isinstance(ret, torch.Tensor):
        >>>         ret = int(ret)
        >>>     else:
        >>>         ret = [int(r) for r in ret]
        >>>         ret = tuple(ret)
        >>>     return ret
    """

    def wrap(func: Callable):
        func_args = dict(func_name=func_name, backend=backend, func=func)
        func_args.update(kwargs)
        func_caller = type(func_name + '@' + backend, (FuncCaller, ),
                           func_args)
        FUNCTION_REWRITER.register_module()(func_caller)
        return func

    return wrap


FUNCTION_REWRITER.register_rewriter = register_rewriter


def apply_rewriter(register_func: Callable) -> Callable:
    """Apply the rewrite function.

    Args:
        register_func (Callable): The registered function.

    Returns:
        Callable: The wrap of registered function.
    """

    def wrapper(*args, **kwargs):
        return register_func(*args, **kwargs)

    return wrapper


class RewriterHook(object):
    """The hook of the rewrite function.

    Args:
        register_name (str): The name of registered rewrite.
        cfg (Dict): Config dictionary of deployment.
    """

    def __init__(self, register_name: str, cfg: Dict, **kwargs):
        func_name, backend = register_name.split('@')
        self.func_name = func_name
        self.backend = backend
        self.register_func = FUNCTION_REWRITER.build(
            func_name, backend=self.backend, cfg=cfg, **kwargs)
        if self.register_func is not None:
            self.origin_func = self.register_func.origin_func
        else:
            self.origin_func = None

    def _set_func(self, rewrite_func: Callable):
        if self.origin_func is not None:
            # import necessary module
            split_path = self.func_name.split('.')
            for i in range(len(split_path), 0, -1):
                try:
                    exec('import {}'.format('.'.join(split_path[:i])))
                    break
                except Exception:
                    continue
            # assign function
            exec(f'{self.func_name} = rewrite_func')

    def __enter__(self):
        self._set_func(apply_rewriter(self.register_func))

    def __exit__(self, type, val, tb):
        self._set_func(self.origin_func)


class RewriterContext(object):
    """The rewrite context.

    The context is used to manage the rewrite functions and the backend.

    Args:
        cfg (Dict): Config dictionary of deployment.
        backend (str): The inference engine name.

    Examples:
        >>> from mmdeploy.core import RewriterContext
        >>> with RewriterContext(cfg, backend='onnxruntime'):
        >>>     # the rewrite has been activated inside the context
        >>>     torch.onnx.export(model, inputs, onnx_file)
    """

    def __init__(self, cfg: Dict, backend: str = 'default', **kwargs):
        self.cfg = cfg
        func_backend_dict = {}
        for register_name in FUNCTION_REWRITER.module_dict:
            register_func, register_backend = register_name.split('@')
            # only build `backend` or `default`
            if register_backend not in [backend, 'default']:
                continue
            if register_func not in func_backend_dict or func_backend_dict[
                    register_func] == 'default':
                func_backend_dict[register_func] = register_backend

        self.hooks = [
            RewriterHook(k + '@' + v, cfg, **kwargs)
            for k, v in func_backend_dict.items()
        ]

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

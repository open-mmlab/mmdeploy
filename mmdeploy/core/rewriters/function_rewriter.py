import logging
from typing import Callable, Dict

from mmcv.utils import Registry

from .register_utils import eval_with_import


# caller wrapper
class FuncCaller(object):
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
    func_caller = registry.module_dict[func_name + '@' + backend]
    assert func_caller is not None, f'{func_name} with {backend} not exist.'
    return func_caller(cfg, **kwargs)


# create register
FUNCTION_REWRITER = Registry('func_rewriters', build_func=build_caller)


# caller decorator
def register_rewriter(func_name: str,
                      backend: str = 'default',
                      **kwargs) -> Callable:

    def wrap(func: Callable):
        func_args = dict(func_name=func_name, backend=backend, func=func)
        func_args.update(kwargs)
        func_caller = type(func_name + '@' + backend, (FuncCaller, ),
                           func_args)
        FUNCTION_REWRITER.register_module()(func_caller)
        return func

    return wrap


FUNCTION_REWRITER.register_rewriter = register_rewriter


def apply_rewriter(regist_func: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        return regist_func(*args, **kwargs)

    return wrapper


class RewriterHook(object):

    def __init__(self, regist_name: str, cfg: Dict, **kwargs):
        func_name, backend = regist_name.split('@')
        self.func_name = func_name
        self.backend = backend
        self.regist_func = FUNCTION_REWRITER.build(
            func_name, backend=self.backend, cfg=cfg, **kwargs)
        if self.regist_func is not None:
            self.origin_func = self.regist_func.origin_func
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
        self._set_func(apply_rewriter(self.regist_func))

    def __exit__(self, type, val, tb):
        self._set_func(self.origin_func)


class RewriterContext(object):

    def __init__(self, cfg: Dict, backend: str = 'default', **kwargs):
        self.cfg = cfg
        func_backend_dict = {}
        for regist_name in FUNCTION_REWRITER.module_dict:
            regist_func, regist_backend = regist_name.split('@')
            # only build `backend` or `default`
            if regist_backend not in [backend, 'default']:
                continue
            if regist_func not in func_backend_dict or func_backend_dict[
                    regist_func] == 'default':
                func_backend_dict[regist_func] = regist_backend

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

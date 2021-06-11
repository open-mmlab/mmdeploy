import logging

from mmcv.utils import Registry
from torch.autograd import Function
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

from .register_utils import eval_with_import


def set_symbolic(cfg, registry, backend='default', opset=11, **kwargs):

    # find valid symbolic
    valid_symbolic_dict = {}
    for module_name, symbolic_impl in registry.module_dict.items():
        func_name, symbolic_backend, is_pytorch = module_name.split('@')
        if symbolic_backend == backend or (symbolic_backend == 'default'
                                           and func_name
                                           not in valid_symbolic_dict):
            valid_symbolic_dict[func_name] = (symbolic_impl,
                                              is_pytorch == 'True')

    # build symbolic
    for func_name in valid_symbolic_dict:
        symbolic_impl, is_pytorch = valid_symbolic_dict[func_name]
        arg_descriptors = symbolic_impl.arg_descriptors
        symbolic_impl = symbolic_impl(cfg=cfg, **kwargs)
        if arg_descriptors is not None and len(arg_descriptors) > 0:
            symbolic_impl = parse_args(*arg_descriptors)(symbolic_impl)
        if is_pytorch:
            register_op(func_name, symbolic_impl, '', opset)
        else:
            try:
                func = eval_with_import(func_name)
                assert issubclass(
                    func,
                    Function), '{} is not an torch.autograd.Function'.format(
                        func_name)
                func.symbolic = symbolic_impl
            except Exception:
                logging.warning(
                    'Can not add symbolic for `{}`'.format(func_name))


SYMBOLICS_REGISTER = Registry('symbolics', build_func=set_symbolic, scope=None)


class SymbolicWrapper:
    func_name = ''
    backend = ''
    is_pytorch = False
    symbolic = None
    arg_descriptors = None

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

    def __call__(self, *args, **kwargs):
        return self.symbolic(*args, **kwargs)


def register_symbolic(func_name,
                      backend='default',
                      is_pytorch=False,
                      arg_descriptors=None,
                      **kwargs):

    def wrapper(symbolic_impl):
        symbolic_args = dict(
            func_name=func_name,
            backend=backend,
            symbolic=symbolic_impl,
            arg_descriptors=arg_descriptors)
        symbolic_args.update(kwargs)
        wrapper_name = '@'.join([func_name, backend, str(is_pytorch)])
        wrapper = type(wrapper_name, (SymbolicWrapper, ), symbolic_args)
        return SYMBOLICS_REGISTER.register_module(wrapper_name)(wrapper)

    return wrapper


SYMBOLICS_REGISTER.register_symbolic = register_symbolic


def register_extra_symbolics(cfg, backend='default', opset=11):
    SYMBOLICS_REGISTER.build(cfg=cfg, backend=backend)

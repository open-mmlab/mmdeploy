# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.utils import IR, Backend, get_root_logger
from .rewriter_utils import (Checker, ContextCaller, RewriterRegistry,
                             copy_function, eval_with_import, get_frame_func,
                             get_func_qualname)


class SymbolicRewriter:
    """A symbolic rewriter which maintains rewritten symbolic.

    The rewritten symbolic can be registered by calling register_symbolic(). In
    RewriteContext, the rewriter automatically registers extra symbolic of
    pytorch and replaces symbolic of custom functions. The symbolic will
    recover after exiting the RewriteContext.

    Examples:
        >>> @SYMBOLIC_REWRITER.register_symbolic('squeeze', \
        >>> is_pytorch=True)
        >>> def squeeze_default(g, self, dim=None):
        >>>     if dim is None:
        >>>         dims = []
        >>>         for i, size in enumerate(self.type().sizes()):
        >>>             if size == 1:
        >>>                 dims.append(i)
        >>>     else:
        >>>         dims = [sym_help._get_const(dim, 'i', 'dim')]
        >>>     return g.op('Squeeze', self, axes_i=dims)
    """

    def __init__(self) -> None:
        self._registry = RewriterRegistry()
        self._func_contexts = defaultdict(list)

    def register_symbolic(self,
                          func_name: str,
                          backend: str = Backend.DEFAULT.value,
                          is_pytorch: bool = False,
                          arg_descriptors: Optional[Sequence[str]] = None,
                          ir: IR = IR.DEFAULT,
                          extra_checkers: Optional[Union[
                              Checker, List[Checker]]] = None,
                          **kwargs) -> Callable:
        """The decorator of the custom symbolic.

        Args:
            func_name (str): The function name/path to override the symbolic.
            backend (str): The rewriter will be activated on which backend.
            is_pytorch (bool): Enable this flag if func_name is the name of \
                a pytorch builtin function.
            arg_descriptors (Sequence[str]): The argument descriptors of the \
                symbol.
            ir (IR): The rewriter will be activated on which IR.
            extra_checkers (Checker | List[Checker] | None): Other requirements
                defined by Checker.

        Returns:
            Callable: The process of registered symbolic.
        """
        return self._registry.register_object(
            func_name,
            backend,
            ir,
            extra_checkers,
            is_pytorch=is_pytorch,
            arg_descriptors=arg_descriptors,
            **kwargs)

    def enter(self,
              cfg: Dict = dict(),
              env: Dict = dict(),
              opset: int = 11,
              **kwargs):
        """The implementation of symbolic register."""
        # clear context
        self._func_contexts.clear()

        # Get current records
        symbolic_records = self._registry.get_records(env)

        self._pytorch_symbolic = list()
        self._extra_symbolic = list()
        new_functions = list()
        for function_name, record_dict in symbolic_records:

            symbolic_function = record_dict['_object']
            symbolic_function = copy_function(symbolic_function)
            arg_descriptors = record_dict['arg_descriptors']
            extra_kwargs = kwargs.copy()
            extra_kwargs.update(record_dict)
            context_caller = ContextCaller(symbolic_function, None, cfg,
                                           **extra_kwargs)

            # register context
            qualname = get_func_qualname(symbolic_function)
            self._func_contexts[qualname].append(context_caller)
            self._func_contexts[function_name].append(context_caller)

            if arg_descriptors is not None and len(arg_descriptors) > 0:
                symbolic_function = parse_args(*arg_descriptors)(
                    symbolic_function)

            is_pytorch = record_dict['is_pytorch']
            if is_pytorch:
                from torch.onnx import register_custom_op_symbolic
                register_custom_op_symbolic(f'::{function_name}',
                                            symbolic_function, opset)

                # Save domain and version
                self._pytorch_symbolic.append((function_name, '', opset))
            else:

                # Check if the origin function exists
                try:
                    origin_func = eval_with_import(function_name)
                    assert issubclass(
                        origin_func,
                        Function), \
                        f'{function_name} is not an torch.autograd.Function'
                except Exception:
                    origin_func = None
                    logger = get_root_logger()
                    logger.warning(
                        f'Can not add symbolic for `{function_name}`')

                # Only register functions that exist
                if origin_func is not None:
                    origin_symbolic = getattr(origin_func, 'symbolic', None)

                    # Save origin function
                    self._extra_symbolic.append((origin_func, origin_symbolic))

                    # Cache new the function to avoid homonymic bug
                    new_functions.append((origin_func, symbolic_function))

            for origin_func, new_func in new_functions:
                origin_symbolic = getattr(origin_func, 'symbolic', None)
                new_func.origin_func = origin_symbolic
                origin_func.symbolic = new_func

    def exit(self):
        """The implementation of symbolic unregister."""
        # clear context
        self._func_contexts.clear()

        # Unregister pytorch op
        if hasattr(torch.onnx, 'unregister_custom_op_symbolic'):
            from torch.onnx import unregister_custom_op_symbolic
            for function_name, domain, version in self._pytorch_symbolic:
                unregister_custom_op_symbolic(f'::{function_name}', version)
        else:
            from torch.onnx.symbolic_registry import \
                _registry as pytorch_registry
            for function_name, domain, version in self._pytorch_symbolic:
                # Same to unregister_op() in torch 1.9.0+
                del pytorch_registry[(domain, version)][function_name]
                if not pytorch_registry[(domain, version)]:
                    del pytorch_registry[(domain, version)]

        # Unregister custom op
        for origin_func, origin_symbolic in self._extra_symbolic:
            origin_func.symbolic = origin_symbolic

    def get_context(self, key: Optional[str] = None) -> ContextCaller:
        """Get the context of rewriter.

        Args:
            key: key to the context.

        Returns:
            ContextCaller: context of function
        """
        func = None
        if key is None:
            func = get_frame_func(2)
            key = get_func_qualname(func)

        # get all contexts
        ctxs = self._func_contexts.get(key, [])

        if func is None:
            assert len(ctxs) == 1
            return ctxs[0]

        ctx = None
        for tmp_ctx in ctxs:
            if tmp_ctx.func == func:
                ctx = tmp_ctx

        if ctx is None:
            get_root_logger().warning(f'Can not found context of {key}')
        return ctx

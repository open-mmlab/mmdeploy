# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional, Sequence

from torch.autograd import Function
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import _registry as pytorch_registry
from torch.onnx.symbolic_registry import register_op

from mmdeploy.utils import Backend, get_root_logger
from .rewriter_utils import ContextCaller, RewriterRegistry, eval_with_import


class SymbolicRewriter:
    """A symbolic rewriter which maintains rewritten symbolic.

    The rewritten symbolic can be registered by calling register_symbolic(). In
    RewriteContext, the rewriter automatically registers extra symbolic of
    pytorch and replaces symbolic of custom functions. The symbolic will
    recover after exiting the RewriteContext.

    Examples:
        >>> @SYMBOLIC_REWRITER.register_symbolic('squeeze', \
        >>> is_pytorch=True)
        >>> def squeeze_default(ctx, g, self, dim=None):
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

    def add_backend(self, backend: str):
        """Add a backend by calling the _registry.add_backend."""
        self._registry.add_backend(backend)

    def register_symbolic(self,
                          func_name: str,
                          backend: str = Backend.DEFAULT.value,
                          is_pytorch: bool = False,
                          arg_descriptors: Optional[Sequence[str]] = None,
                          **kwargs) -> Callable:
        """The decorator of the custom symbolic.

        Args:
            func_name (str): The function name/path to override the symbolic.
            backend (str): The inference engine name.
            is_pytorch (bool): Enable this flag if func_name is the name of \
                a pytorch builtin function.
            arg_descriptors (Sequence[str]): The argument descriptors of the \
                symbol.

        Returns:
            Callable: The process of registered symbolic.
        """
        return self._registry.register_object(
            func_name,
            backend,
            is_pytorch=is_pytorch,
            arg_descriptors=arg_descriptors,
            **kwargs)

    def enter(self,
              cfg: Dict = dict(),
              backend: str = Backend.DEFAULT.value,
              opset: int = 11,
              **kwargs):
        """The implementation of symbolic register."""
        # Get current records
        symbolic_records = self._registry.get_records(backend)

        self._pytorch_symbolic = list()
        self._extra_symbolic = list()
        new_functions = list()
        for function_name, record_dict in symbolic_records:

            symbolic_function = record_dict['_object']
            arg_descriptors = record_dict['arg_descriptors']
            extra_kwargs = kwargs.copy()
            extra_kwargs.update(record_dict)
            context_caller = ContextCaller(symbolic_function, None, cfg,
                                           **extra_kwargs)
            if arg_descriptors is not None and len(arg_descriptors) > 0:
                context_caller = parse_args(*arg_descriptors)(context_caller)

            is_pytorch = record_dict['is_pytorch']
            if is_pytorch:
                register_op(function_name, context_caller, '', opset)

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
                    new_functions.append((origin_func, context_caller))

            for origin_func, new_func in new_functions:
                origin_symbolic = getattr(origin_func, 'symbolic', None)
                new_func.origin_func = origin_symbolic
                origin_func.symbolic = new_func

    def exit(self):
        """The implementation of symbolic unregister."""
        # Unregister pytorch op
        for function_name, domain, version in self._pytorch_symbolic:
            # Same to ungister_op() in torch 1.9.0+
            del pytorch_registry[(domain, version)][function_name]
            if not pytorch_registry[(domain, version)]:
                del pytorch_registry[(domain, version)]

        # Unregister custom op
        for origin_func, origin_symbolic in self._extra_symbolic:
            origin_func.symbolic = origin_symbolic

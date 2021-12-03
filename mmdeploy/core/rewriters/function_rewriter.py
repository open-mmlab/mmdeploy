# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Callable, Dict

from mmdeploy.utils.constants import Backend
from .rewriter_utils import ContextCaller, RewriterRegistry, eval_with_import


def _set_func(origin_func_name: str, rewrite_func: Callable):
    """Rewrite a function by executing a python statement."""

    # Import necessary module
    split_path = origin_func_name.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue
    # Assign function
    exec(f'{origin_func_name} = rewrite_func')


class FunctionRewriter:
    """A function rewriter which maintains rewritten functions.

    The rewritten functions can be registered by calling register_rewriter().
    In RewriteContext, the rewriter automatically replaces target functions and
    recovers them after exiting the context.

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

    def __init__(self):
        self._registry = RewriterRegistry()

    def add_backend(self, backend: str):
        """Add a backend by calling the _registry.add_backend."""
        self._registry.add_backend(backend)

    def register_rewriter(self,
                          func_name: str,
                          backend: str = Backend.DEFAULT.value,
                          **kwargs):
        """The interface of function rewriter decorator.

        Args:
            func_name (str): The function name/path to rewrite.
            backend (str): The inference engine name.
        Returns:
            Callable: The process of registering function.
        """

        return self._registry.register_object(func_name, backend, **kwargs)

    def enter(self,
              cfg: Dict = dict(),
              backend: str = Backend.DEFAULT.value,
              **kwargs):
        """The implementation of function rewrite."""
        # Get current records
        functions_records = self._registry.get_records(backend)

        self._origin_functions = list()
        new_functions = list()
        for function_name, record_dict in functions_records:

            # Check if the origin function exists
            try:
                origin_func = eval_with_import(function_name)
            except Exception:
                origin_func = None
                logging.warning(
                    f'Can not find {function_name}, function rewrite will '
                    'not be applied')

            # Only rewrite functions that exist
            if origin_func is not None:

                # Save origin function
                self._origin_functions.append((function_name, origin_func))

                # Create context_caller
                rewrite_function = record_dict['_object']
                extra_kwargs = kwargs.copy()
                extra_kwargs.update(record_dict)
                context_caller = ContextCaller(
                    rewrite_function, origin_func, cfg,
                    **extra_kwargs).get_wrapped_caller()

                # Cache new the function to avoid homonymic bug
                new_functions.append((function_name, context_caller))

        for function_name, new_function in new_functions:
            # Rewrite functions
            _set_func(function_name, new_function)

    def exit(self):
        """Recover the function rewrite."""
        for func_name, func in self._origin_functions:
            _set_func(func_name, func)

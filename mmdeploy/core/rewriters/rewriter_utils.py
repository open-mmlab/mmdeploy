# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from mmdeploy.utils.constants import Backend


def eval_with_import(path: str) -> Any:
    """Evaluate the string as Python script.

    Args:
        path (str): The path to evaluate.

    Returns:
        Any: The result of evaluation.
    """
    split_path = path.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue
    return eval(path)


def import_function(path: str) -> Tuple[Callable, Optional[type]]:
    """Import and evaluate a function. If the function is defined in a class,
    evaluate the class additionally.

    Args:
        path (str): The path to evaluate.

    Returns:
        Callable: The function of evaluation.
        type: The class of evaluation if the function is defined in a class, or
            None.
    """
    split_path = path.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue

    obj = eval(path)

    # The path that might be a class
    previous_obj = eval('.'.join(split_path[:-1]))

    # Check if the path leads to a class
    if inspect.isclass(previous_obj):
        return obj, previous_obj
    else:
        return obj, None


class RewriterRegistry:
    """A registry that recoreds rewrite objects.

    Logically this class is a two-dimensional table which maintains an object
    list for each backend. The records can be inserted to this table through
    register.

    Members:
        _rewrite_records (Dict[Backend, Dict[str, Dict]]): A data structure
            which records the register message in a specific backend.

    Example:
        >>> FUNCTION_REGISTRY = RewriterRegistry()
        >>> @FUNCTION_REGISTRY.register_object(backend="default")
        >>> def add():
        >>>     return a + b
        >>> records = FUNCTION_REGISTRY.get_record("default")
    """

    # TODO: replace backend string with "Backend" constant
    def __init__(self):
        self._rewrite_records = dict()
        self.add_backend(Backend.DEFAULT.value)

    def _check_backend(self, backend: str):
        """Check if a backend has been supported."""
        if backend not in self._rewrite_records:
            raise Exception('Backend is not supported by registry.')

    def add_backend(self, backend: str):
        """Add a backend dictionary."""
        if backend not in self._rewrite_records:
            self._rewrite_records[backend] = dict()

    def get_records(self, backend: str) -> List:
        """Get all registered records in record table."""
        self._check_backend(backend)

        if backend != Backend.DEFAULT.value:
            # Update dict A with dict B.
            # Then convert the result dict to a list, while keeping the order
            # of A and B: the elements only belong to B should alwarys come
            # after the elements only belong to A.
            # The complexity is O(n + m).
            dict_a = self._rewrite_records[Backend.DEFAULT.value]
            dict_b = self._rewrite_records[backend]
            records = []
            for k, v in dict_a.items():
                if k in dict_b:
                    records.append((k, dict_b[k]))
                else:
                    records.append((k, v))
            for k, v in dict_b.items():
                if k not in dict_a:
                    records.append((k, v))
        else:
            records = list(
                self._rewrite_records[Backend.DEFAULT.value].items())
        return records

    def _register(self, name: str, backend: str, **kwargs):
        """The implementation of register."""
        self._check_backend(backend)
        self._rewrite_records[backend][name] = kwargs

    def register_object(self, name: str, backend: str, **kwargs) -> Callable:
        """The decorator to register an object."""
        self._check_backend(backend)

        def decorator(object):
            self._register(name, backend, _object=object, **kwargs)
            return object

        return decorator


class ContextCaller:
    """A callable object used in RewriteContext.

    This class saves context variables as member variables. When a rewritten
    function is called in RewriteContext, an instance of this class will be
    passed as the first argument of the function.

    Args:
        func (Callable): The rewritten function to call.
        origin_func (Callable): The function that is going to be rewritten.
            Note that in symbolic function origin_func may be 'None'.
        cfg (Dict): The deploy config dictionary.

    Example:
        >>> @FUNCTION_REWRITER.register_rewriter(func_name='torch.add')
        >>> def func(ctx, x, y):
        >>>     # ctx is an instance of ContextCaller
        >>>     print(ctx.cfg)
        >>>     return x + y
    """

    def __init__(self, func: Callable, origin_func: Callable, cfg: Dict,
                 **kwargs):
        self.func = func
        self.origin_func = origin_func
        self.cfg = cfg
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        """Directly call self.func."""
        return self.func(self, *args, **kwargs)

    def get_wrapped_caller(self):
        """Generate a wrapped caller for function rewrite."""

        # Rewrite function should not call a member function, so we use a
        # wrapper to generate a Callable object.
        def wrapper(*args, **kwargs):
            # Add a new argument (context message) to function
            # Because "self.func" is a function but not a member function,
            # we should pass self as the first argument
            return self.func(self, *args, **kwargs)

        return wrapper

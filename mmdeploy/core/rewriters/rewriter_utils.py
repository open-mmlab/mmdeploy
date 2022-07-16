# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mmdeploy
from mmdeploy.utils.constants import IR, Backend


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


def collect_env(backend: Backend, ir: IR, **kwargs) -> Dict:
    """Collect current environment information, including backend, ir, codebase
    version, etc. Rewriters will be checked according to env infos.

    Args:
        backend (Backend): Current backend.
        ir (IR): Current IR.

    Returns:
        Dict: Record the value of Backend and IR as well as the versions of
        libraries.
    """
    from mmdeploy.utils import get_backend_version, get_codebase_version
    env = dict(backend=backend, ir=ir)
    env['mmdeploy'] = mmdeploy.__version__
    env.update(get_backend_version())
    env.update(get_codebase_version())
    env.update(kwargs)
    return env


class Checker(metaclass=ABCMeta):
    """The interface for checking whether a rewriter is valid."""

    def __init__(self):
        pass

    @abstractmethod
    def check(self, env: Dict) -> bool:
        """Check the if the rewriter is valid according to environment.

        Args:
            env (Dict): The backend, IR info and version info.
        """
        pass


class BackendChecker(Checker):
    """Checker that determines which backend the rewriter must run on.

    Args:
        required_backend (Backend): The rewriter will be activated on
            which backend.
    """

    def __init__(self, required_backend: Backend):
        super().__init__()
        self.required_backend = required_backend

    def check(self, env: Dict) -> bool:
        """Check the if the rewriter is valid according to backend.

        Args:
            env (Dict): The backend, IR info and version info.
        """
        return env['backend'] == self.required_backend


class IRChecker(Checker):
    """Checker that determines which IR the rewriter must run on.

    Args:
        required_ir (IR): The rewriter will be activated on which IR.
    """

    def __init__(self, required_ir: IR):
        super().__init__()
        self.required_ir = required_ir

    def check(self, env: Dict) -> bool:
        """Check the if the rewriter is valid according to IR.

        Args:
            env (Dict): The backend, IR info and version info.
        """
        return env['ir'] == self.required_ir


class LibVersionChecker(Checker):
    """Checker that determines which IR the rewriter must run on.

    Args:
        lib (str): The name of library.
        min_version (str | None): The rewriter should no lower than which
            version. Default to `None`.
        max_version (str | None): The rewriter should no greater than which
            version. Default to `None`.
    """

    def __init__(self,
                 lib: str,
                 min_version: Optional[str] = None,
                 max_version: Optional[str] = None):
        super().__init__()
        self.lib = lib
        self.min_version = min_version
        self.max_version = max_version

    def check(self, env: Dict) -> bool:
        """Check the if the rewriter is valid according to library version.

        Args:
            env (Dict): The backend, IR info and version info.
        """
        # If the library has not been installed
        if env[self.lib] is None:
            return False

        from packaging import version
        valid = True
        # The version should no less than min version and no greater than
        # max version.
        if self.min_version is not None:
            if version.parse(env[self.lib]) < version.parse(self.min_version):
                valid = False
        if self.max_version is not None:
            if version.parse(env[self.lib]) > version.parse(self.max_version):
                valid = False
        return valid


class RewriterRegistry:
    """A registry that records rewrite objects.

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

    def __init__(self):
        self._rewrite_records = dict()

    def get_records(self, env: Dict) -> List:
        """Get all registered records that are valid in the given environment
        from record table.

        If the backend and IR of rewriter are set to 'default', then the
        rewriter is regarded as default rewriter. The default rewriter will be
        activated only when all other rewriters are not valid. If there are
        multiple rewriters are valid (except default rewriter), we will
        activate the first one (The order is determined by the time when
        rewriters are loaded).

        Args:
            env (dict): Environment dictionary that includes backend, IR,
                codebase version, etc.

        Returns:
            List: A list that includes valid records.
        """
        default_records = list()
        records = list()

        for origin_function, rewriter_records in self._rewrite_records.items():
            default_rewriter = None
            final_rewriter = None
            for record in rewriter_records:
                # Get the checkers of current rewriter
                checkers: List[Checker] = record['_checkers']

                # Check if the rewriter is default rewriter
                if len(checkers) == 0:
                    #  Process the default rewriter exceptionally
                    if default_rewriter is None:
                        default_rewriter = record
                    else:
                        warnings.warn(
                            'Detect multiple valid rewriters for'
                            f'{origin_function}, use the first rewriter.')
                else:
                    # Check if the checker is valid.
                    # The checker is valid only if all the checks are passed
                    valid = True
                    for checker in checkers:
                        if not checker.check(env):
                            valid = False
                            break

                    if valid:
                        # Check if there are multiple valid rewriters
                        if final_rewriter is not None:
                            warnings.warn(
                                'Detect multiple valid rewriters for'
                                f'{origin_function}, use the first rewriter.')
                        else:
                            final_rewriter = record

            # Append final rewriter.
            # If there is no valid rewriter, try not apply default rewriter
            if final_rewriter is not None:
                records.append((origin_function, final_rewriter))
            elif default_rewriter is not None:
                default_records.append((origin_function, default_rewriter))

        # Make the default records como to the front of list because we may
        # want the non-default records to override them.
        return default_records + records

    def _register(self, name: str, backend: Backend, ir: IR,
                  extra_checkers: List[Checker], **kwargs):
        """The implementation of register."""

        # Merge checkers to kwargs
        record_dict = kwargs

        # Try to create a checker according to 'backend' field
        if backend != Backend.DEFAULT:
            extra_checkers.append(BackendChecker(backend))

        # Try to create a checker according to 'ir' field
        if ir != IR.DEFAULT:
            extra_checkers.append(IRChecker(ir))

        record_dict['_checkers'] = extra_checkers

        # There may be multiple rewriters of a function/module. We use a list
        # to store the rewriters of a function/module.
        if name not in self._rewrite_records:
            self._rewrite_records[name] = list()
        self._rewrite_records[name].append(record_dict)

    def register_object(self,
                        name: str,
                        backend: str,
                        ir: IR,
                        extra_checkers: Optional[Union[Checker,
                                                       List[Checker]]] = None,
                        **kwargs) -> Callable:
        """The decorator to register an object.

        Args:
            name (str): The import path to access the function/module.
            backend (str): The rewriter will be activated on which backend.
            ir (IR): The rewriter will be activated on which ir.
            extra_checkers (None | Checker | List[Checker]): Other requirements
                for the rewriters. Default to `None`.

        Returns:
            Callable: The decorator.
        """

        if extra_checkers is None:
            extra_checkers = []
        elif isinstance(extra_checkers, Checker):
            extra_checkers = [extra_checkers]

        backend = Backend.get(backend)

        def decorator(object):
            self._register(
                name, backend, ir, extra_checkers, _object=object, **kwargs)
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
        # PyTorch will do annotation check on symbolic function
        # Update the annotation so ContextCaller can pass the check.
        if origin_func is not None:
            wraps(origin_func)(self)
        else:
            self.__annotations__ = getattr(func, '__annotations__', {})

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

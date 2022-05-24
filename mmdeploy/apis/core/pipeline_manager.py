# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from mmdeploy.utils import get_root_logger

try:
    import torch.multiprocessing as mp
except Exception:
    import multiprocessing as mp


def _get_func_name(func: Callable) -> str:
    """get function name."""
    assert isinstance(func, Callable), f'{func} is not a Callable object.'
    _func_name = None
    if hasattr(func, '__qualname__'):
        _func_name = f'{func.__module__}.{func.__qualname__}'
    elif hasattr(func, '__class__'):
        _func_name = func.__class__
    else:
        _func_name = str(func)
    return _func_name


class PipelineCaller:
    """Classes to record the attribute of each pipeline function."""

    def __init__(self,
                 module_name: str,
                 impl_name: str,
                 func_name: Optional[str] = None,
                 log_level: int = logging.DEBUG,
                 is_multiprocess_available: bool = True) -> None:
        if func_name is not None:
            self._func_name = func_name
        else:
            self._func_name = impl_name
        # Can not save the function directly since multiprocess with spawn mode
        # require all field can be pickled.
        self._module_name = module_name
        self._impl_name = impl_name
        self._is_multiprocess_available = is_multiprocess_available
        self._enable_multiprocess = False
        self._mp_dict = None
        self._mp_async = False
        self._call_id = 0
        self._log_level = log_level
        self._input_hooks: List[Callable] = []
        self._output_hooks: List[Callable] = []

    @property
    def is_multiprocess_available(self) -> bool:
        """check if multiprocess is available for this pipeline."""
        return self._is_multiprocess_available

    @property
    def is_multiprocess(self) -> bool:
        """check if this pipeline is multiprocess."""
        return self._enable_multiprocess

    @property
    def input_hooks(self) -> List[Callable]:
        """get input hooks."""
        return self._input_hooks

    @property
    def output_hooks(self) -> List[Callable]:
        """get output hooks."""
        return self._output_hooks

    def pop_mp_output(self, call_id: int = None) -> Any:
        """pop multiprocess output."""
        assert self._mp_dict is not None, 'mp_dict is None.'
        call_id = self._call_id if call_id is None else call_id
        assert call_id in self._mp_dict, \
            f'`{self._func_name}` with Call id: {call_id} failed.'
        ret = self._mp_dict[call_id]
        self._mp_dict.pop(call_id)
        return ret

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        do_multiprocess = self.is_multiprocess_available \
            and self.is_multiprocess\
            and self._mp_dict is not None

        logger = get_root_logger(log_level=self._log_level)
        mp_log_str = 'subprocess' if do_multiprocess else 'main process'
        logger.log(self._log_level,
                   f'Start pipeline {self._func_name} in {mp_log_str}')

        for input_hook in self.input_hooks:
            args, kwargs = input_hook(*args, **kwargs)

        module_name = self._module_name
        impl_name = self._impl_name
        # TODO: find another way to load function
        mod = importlib.import_module(module_name)
        func = getattr(mod, impl_name, None)
        assert func is not None, \
            f'Can not find implementation of {self._func_name}'
        ret = func(*args, **kwargs)
        for output_hook in self.output_hooks:
            ret = output_hook(ret)

        if do_multiprocess:
            self._mp_dict[self._call_id] = ret

        logger.log(self._log_level, f'Finish pipeline {self._func_name}')
        return ret


class PipelineResult:
    """The result of async pipeline."""

    def __init__(self, manager: Any, call_id: int) -> None:
        self._manager = manager
        self._call_id = call_id

    @property
    def call_id(self) -> int:
        return self._call_id

    def get(self) -> Any:
        """get result."""
        return self._manager.get_result_sync(self._call_id)


FUNC_NAME_TYPE = Union[str, Callable]


class PipelineManager:
    """This is a tool to manager all pipeline functions."""

    def __init__(self) -> None:
        self._enable_multiprocess = True
        self._mp_manager = None
        self._callers: Dict[str, PipelineCaller] = dict()
        self._call_id = 0
        self._proc_async: Dict[int, (str, mp.Process)] = dict()

    @property
    def mp_manager(self) -> Optional[mp.Manager]:
        """get multiprocess manager."""
        return self._mp_manager

    def get_caller(self, func_name: FUNC_NAME_TYPE) -> PipelineCaller:
        """get caller of given function."""
        if isinstance(func_name, Callable):
            func_name = _get_func_name(func_name)
        assert func_name in self._callers, \
            f'{func_name} has not been registered.'
        return self._callers[func_name]

    def __set_caller_val(self,
                         val_name: str,
                         val: Any,
                         func_name: Optional[FUNC_NAME_TYPE] = None) -> None:
        """helper to set any caller value."""
        if func_name is None:
            for func_name_ in self._callers:
                setattr(self.get_caller(func_name_), val_name, val)
        else:
            setattr(self.get_caller(func_name), val_name, val)

    def _create_mp_manager(self) -> None:
        """create multiprocess manager if not exists."""
        if self._mp_manager is None:
            self._mp_manager = mp.Manager()

    def _enable_multiprocess_single(self,
                                    val: bool,
                                    func_name: FUNC_NAME_TYPE = None) -> None:
        """implement of enable_multiprocess."""
        pipe_caller = self.get_caller(func_name)
        # check if multiprocess is available for this function
        if not pipe_caller.is_multiprocess_available:
            return
        pipe_caller._enable_multiprocess = val
        if val is True and self.mp_manager is not None:
            pipe_caller._mp_dict = self.mp_manager.dict()
        else:
            pipe_caller._mp_dict = None

    def enable_multiprocess(
        self,
        val: bool,
        func_names: Optional[Union[FUNC_NAME_TYPE,
                                   Sequence[FUNC_NAME_TYPE]]] = None
    ) -> None:
        """enable multiprocess for pipeline function.

        Args:
            val (bool): enable or disable multiprocess.
            func_names (str | List[str]): function names to enable. If
                func_name is None, all registered function will be enabled.
        """
        if val is True:
            self._create_mp_manager()
        if func_names is None:
            for func_name in self._callers:
                self._enable_multiprocess_single(val, func_name=func_name)
        else:
            if isinstance(func_names, str):
                func_names = [func_names]
            for func_name in func_names:
                self._enable_multiprocess_single(val, func_name=func_name)

    def set_mp_async(self,
                     val: bool,
                     func_name: Optional[FUNC_NAME_TYPE] = None) -> None:
        """set multiprocess async of the pipeline function.

        Args:
            val (bool): enable async call.
            func_name (str | None): function name to set. If func_name is
                None, all registered function will be set.
        """
        self.__set_caller_val('_mp_async', val, func_name)

    def set_log_level(
        self,
        level: int,
        func_names: Optional[Union[FUNC_NAME_TYPE,
                                   Sequence[FUNC_NAME_TYPE]]] = None
    ) -> None:
        """set log level of the pipeline function.

        Args:
            level (int): the log level.
            func_names (str | List[str]): function names to set. If func_names
                is None, all registered function will be set.
        """
        if isinstance(func_names, str):
            func_names = [func_names]
        for func_name in func_names:
            self.__set_caller_val('_log_level', level, func_name)

    def get_input_hooks(self, func_name: FUNC_NAME_TYPE):
        """get input hooks of given function name.

        Args:
            func_name (str): function name.
        """
        pipe_caller = self.get_caller(func_name)
        return pipe_caller.input_hooks

    def get_output_hooks(self, func_name: FUNC_NAME_TYPE):
        """get output hooks of given function name.

        Args:
            func_name (str): function name.
        """
        pipe_caller = self.get_caller(func_name)
        return pipe_caller.output_hooks

    def call_function_local(self, func_name: FUNC_NAME_TYPE, *args,
                            **kwargs) -> Any:
        """call pipeline function.

        Args:
            func_name (str): function name to be called.

        Returns:
            Any: The result of call function
        """
        pipe_caller = self.get_caller(func_name)
        pipe_caller._call_id = self._call_id
        self._call_id += 1
        return pipe_caller(*args, **kwargs)

    def call_function_async(self, func_name: FUNC_NAME_TYPE, *args,
                            **kwargs) -> int:
        """call pipeline function.

        Args:
            func_name (str): function name to be called.

        Returns:
            int: Call id of this function
        """
        pipe_caller = self.get_caller(func_name)
        assert pipe_caller.is_multiprocess, \
            f'multiprocess of {func_name} has not been enabled.'

        call_id = self._call_id
        pipe_caller._call_id = call_id
        self._call_id += 1
        proc = mp.Process(target=pipe_caller, args=args, kwargs=kwargs)
        proc.start()
        self._proc_async[call_id] = (func_name, proc)

        return call_id

    def get_result_sync(self, call_id: int):
        """get result of async call."""
        assert call_id in self._proc_async, f'Unknown call id: {call_id}'
        func_name, proc = self._proc_async.pop(call_id)
        proc.join()
        ret = self.get_caller(func_name).pop_mp_output(call_id)

        return ret

    def call_function(self, func_name: FUNC_NAME_TYPE, *args, **kwargs) -> Any:
        """call pipeline function.

        Args:
            func_name (str): function name to be called.

        Returns:
            Any: The result of call function
        """
        pipe_caller = self.get_caller(func_name)

        if self._enable_multiprocess and pipe_caller.is_multiprocess:
            call_id = self.call_function_async(func_name, *args, **kwargs)
            if pipe_caller._mp_async:
                return PipelineResult(self, call_id)
            return self.get_result_sync(call_id)
        else:
            return self.call_function_local(func_name, *args, **kwargs)

    def register_pipeline(self,
                          is_multiprocess_available: bool = True,
                          log_level: int = logging.DEBUG):
        """register the pipeline function."""

        def _register(func):
            assert isinstance(func, Callable), f'{func} is not Callable.'
            func_name_ = _get_func_name(func)

            # save the implementation into the registry module
            impl_name = f'_pipe_{func.__name__}__impl_'
            frame = inspect.stack()[1]
            outer_mod = inspect.getmodule(frame[0])
            mod_name = outer_mod.__name__
            setattr(outer_mod, impl_name, func)

            # create caller
            pipe_caller = PipelineCaller(
                mod_name,
                impl_name,
                func_name=func_name_,
                log_level=log_level,
                is_multiprocess_available=is_multiprocess_available)
            PIPELINE_MANAGER._callers[func_name_] = pipe_caller

            # wrap call
            @wraps(func)
            def _wrap(*args, **kwargs):
                return self.call_function(func_name_, *args, **kwargs)

            return _wrap

        return _register


PIPELINE_MANAGER = PipelineManager()


class no_mp:
    """The context manager used to disable multiprocess."""

    def __init__(self, manager: PipelineManager = PIPELINE_MANAGER) -> None:
        self._manager = manager
        self._old_enable_multiprocess = True

    def __enter__(self):
        self._old_enable_multiprocess = self._manager._enable_multiprocess
        self._manager._enable_multiprocess = False

    def __exit__(self, type, val, tb):
        self._manager._enable_multiprocess = self._old_enable_multiprocess

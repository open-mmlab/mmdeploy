# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from mmdeploy.utils import get_root_logger

try:
    import torch.multiprocessing as mp
except Exception:
    import multiprocessing as mp


def __get_func_name(func: Callable) -> str:
    """get function name."""
    assert isinstance(func, Callable), f'{func} is not a Callable object.'
    _func_name = None
    if hasattr(func, '__name__'):
        _func_name = func.__name__
    elif hasattr(func, '__class__'):
        _func_name = func.__class__
    else:
        _func_name = str(func)
    return _func_name


class PipelineWrapper:
    """Classes to record the attribute of each pipeline function."""

    def __init__(self,
                 func: Callable,
                 func_name: Optional[str] = None,
                 is_multiprocess_available: bool = True) -> None:
        assert isinstance(func, Callable)
        if func_name is not None:
            self._func_name = func_name
        else:
            self._func_name = __get_func_name(func)
        self._func = func
        self._is_multiprocess_available = is_multiprocess_available
        self._enable_multiprocess = False
        self._mp_dict = None
        self._call_id = 0
        self._log_level = logging.INFO
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
            f'mp output of {self._func_name} ' \
            f'with call id:{call_id} is not exist.'
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

        ret = self._func(*args, **kwargs)
        for output_hook in self.output_hooks:
            ret = output_hook(ret)

        if do_multiprocess:
            self._mp_dict[self._call_id] = ret

        logger.log(self._log_level, f'Finish pipeline {self._func_name}')
        return ret


class PipelineManager:
    """This is a tool to manager all pipeline functions."""

    def __init__(self) -> None:
        self._enable_multiprocess = False
        self._mp_manager = None
        self._pipelines: Dict[str, PipelineWrapper] = dict()
        self._call_id = 0

    @property
    def mp_manager(self) -> Optional[mp.Manager]:
        """get multiprocess manager."""
        return self._mp_manager

    def _create_mp_manager(self) -> None:
        """create multiprocess manager if not exists."""
        if self._mp_manager is None:
            self._mp_manager = mp.Manager()

    def _enable_multiprocess_single(self,
                                    val: bool,
                                    func_name: str = None) -> None:
        """implement of enable_multiprocess."""
        pipe_wrapper = self._pipelines[func_name]
        # check if multiprocess is available for this function
        if not pipe_wrapper.is_multiprocess_available:
            return
        pipe_wrapper._enable_multiprocess = val
        if val is True and self.mp_manager is not None:
            pipe_wrapper._mp_dict = self.mp_manager.dict()
        else:
            pipe_wrapper._mp_dict = None

    def enable_multiprocess(self,
                            val: bool,
                            func_name: Optional[str] = None) -> None:
        """enable multiprocess for pipeline function.

        Args:
            val (bool): enable or disable multiprocess.
            func_name (str | None): function name to enable. If func_name is
                None, all registered function will be enabled.
        """
        if val is True:
            self._create_mp_manager()
        if func_name is None:
            for func_name in self._pipelines:
                self._enable_multiprocess_single(val, func_name=func_name)
        else:
            self._enable_multiprocess_single(val, func_name=func_name)

    def set_log_level(self,
                      level: int,
                      func_name: Optional[str] = None) -> None:
        """set log level of the pipeline function.

        Args:
            level (int): the log level.
            func_name (str | None): function name to set. If func_name is
                None, all registered function will be set.
        """
        if func_name is None:
            for func_name_ in self._pipelines:
                self._pipelines[func_name_]._log_level = level
        else:
            self._pipelines[func_name]._log_level = level

    def get_input_hooks(self, func_name: str):
        """get input hooks of given function name.

        Args:
            func_name (str): function name.
        """
        assert func_name in self._pipelines, \
            f'{func_name} as not been registered.'
        pipe_wrapper = self._pipelines[func_name]
        return pipe_wrapper.input_hooks

    def get_output_hooks(self, func_name: str):
        """get output hooks of given function name.

        Args:
            func_name (str): function name.
        """
        assert func_name in self._pipelines, \
            f'{func_name} as not been registered.'
        pipe_wrapper = self._pipelines[func_name]
        return pipe_wrapper.output_hooks

    def call_function(self, func_name: str, *args, **kwargs) -> Any:
        """call pipeline function.

        Args:
            func_name (str): function name to be called.
        """
        assert func_name in self._pipelines, \
            f'{func_name} as not been registered.'
        pipe_wrapper = self._pipelines[func_name]
        pipe_wrapper._call_id = self._call_id
        self._call_id += 1
        if pipe_wrapper.is_multiprocess:
            proc = mp.Process(target=pipe_wrapper, args=args, kwargs=kwargs)
            proc.start()
            proc.join()

            ret = pipe_wrapper.pop_mp_output()
            return ret
        else:
            return pipe_wrapper(*args, **kwargs)

    def register_pipeline(self,
                          func_name: str = None,
                          is_multiprocess_available: bool = True):
        """register the pipeline function."""

        def _register(func):
            assert isinstance(func, Callable), f'{func} is not Callable.'
            func_name_ = func_name if func_name is not None \
                else __get_func_name(func)
            pipe_wrapper = PipelineWrapper(
                func,
                func_name=func_name_,
                is_multiprocess_available=is_multiprocess_available)
            PIPELINE_MANAGER._pipelines[func_name_] = pipe_wrapper

            @wraps(func)
            def _wrap(*args, **kwargs):
                return self.call_function(func_name_, *args, **kwargs)

            return _wrap

        return _register


PIPELINE_MANAGER = PipelineManager()

# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
from mmcv.utils import get_logger


class TimeCounter:
    """A tool for counting inference time of backends."""
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls,
                   warmup: int = 1,
                   log_interval: int = 1,
                   with_sync: bool = False):
        """Proceed time counting.

        Args:
            warmup (int): The warm up steps, default 1.
            log_interval (int): Interval between each log, default 1.
            with_sync (bool): Whether use cuda synchronize for time counting,
                default `False`.
        """

        def _register(func):
            assert warmup >= 1
            assert func.__name__ not in cls.names,\
                'The registered function name cannot be repeated!'
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(
                count=0,
                execute_time=0,
                log_interval=log_interval,
                warmup=warmup,
                with_sync=with_sync,
                enable=False)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                execute_time = cls.names[func.__name__]['execute_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup = cls.names[func.__name__]['warmup']
                with_sync = cls.names[func.__name__]['with_sync']
                enable = cls.names[func.__name__]['enable']

                count += 1
                cls.names[func.__name__]['count'] = count

                if enable:
                    if with_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if enable:
                    if with_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time

                if enable and count > warmup:
                    execute_time += elapsed
                    cls.names[func.__name__]['execute_time'] = execute_time

                    if (count - warmup) % log_interval == 0:
                        times_per_count = 1000 * execute_time / (
                            count - warmup)
                        msg = f'[{func.__name__}]-{count} times per count: '\
                              f'{times_per_count:.2f} ms, '\
                              f'{1000/times_per_count:.2f} FPS'
                        cls.logger.info(msg)

                return result

            return fun

        return _register

    @classmethod
    @contextmanager
    def activate(cls,
                 func_name: str = None,
                 warmup: int = 1,
                 log_interval: int = 1,
                 with_sync: bool = False,
                 file: Optional[str] = None):
        """Activate the time counter.

        Args:
            func_name (str): Specify which function to activate. If not
                specified, all registered function will be activated.
            warmup (int): the warm up steps, default 1.
            log_interval (int): Interval between each log, default 1.
            with_sync (bool): Whether use cuda synchronize for time counting,
                default False.
            file (str | None): The file to save output messages. The default
                is `None`.
        """
        assert warmup >= 1
        logger = get_logger('test', log_file=file)
        cls.logger = logger
        if func_name is not None:
            warnings.warn('func_name must be globally unique if you call '
                          'activate multiple times')
            assert func_name in cls.names, '{} must be registered before '\
                'setting params'.format(func_name)
            cls.names[func_name]['warmup'] = warmup
            cls.names[func_name]['log_interval'] = log_interval
            cls.names[func_name]['with_sync'] = with_sync
            cls.names[func_name]['enable'] = True
        else:
            for name in cls.names:
                cls.names[name]['warmup'] = warmup
                cls.names[name]['log_interval'] = log_interval
                cls.names[name]['with_sync'] = with_sync
                cls.names[name]['enable'] = True
        yield
        if func_name is not None:
            cls.names[func_name]['enable'] = False
        else:
            for name in cls.names:
                cls.names[name]['enable'] = False

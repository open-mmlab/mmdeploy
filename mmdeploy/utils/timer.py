# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
from contextlib import contextmanager
from logging import Logger
from typing import Optional

import numpy as np
import torch

from mmdeploy.utils.logging import get_logger


class TimeCounter:
    """A tool for counting inference time of backends."""
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls,
                   name: str,
                   warmup: int = 1,
                   log_interval: int = 1,
                   with_sync: bool = False):
        """Proceed time counting.

        Args:
            name (str): Name of this timer.
            warmup (int): The warm up steps, default 1.
            log_interval (int): Interval between each log, default 1.
            with_sync (bool): Whether use cuda synchronize for time counting,
                default `False`.
        """

        def _register(func):
            assert warmup >= 1
            assert name not in cls.names,\
                'The registered function name cannot be repeated!'
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[name] = dict(
                count=0,
                execute_time=[],
                log_interval=log_interval,
                warmup=warmup,
                with_sync=with_sync,
                batch_size=1,
                enable=False)

            def fun(*args, **kwargs):
                count = cls.names[name]['count']
                execute_time = cls.names[name]['execute_time']
                log_interval = cls.names[name]['log_interval']
                warmup = cls.names[name]['warmup']
                with_sync = cls.names[name]['with_sync']
                batch_size = cls.names[name]['batch_size']
                enable = cls.names[name]['enable']

                count += 1
                cls.names[name]['count'] = count

                if enable:
                    if with_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if enable:
                    if with_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start_time) / batch_size

                if enable and count > warmup:
                    execute_time.append(elapsed)

                    if (count - warmup) % log_interval == 0:
                        times_per_count = 1000 * float(np.mean(execute_time))
                        fps = 1000 / times_per_count
                        msg = f'[{name}]-{count} times per count: '\
                              f'{times_per_count:.2f} ms, '\
                              f'{fps:.2f} FPS'
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
                 file: Optional[str] = None,
                 logger: Optional[Logger] = None,
                 batch_size: int = 1,
                 **kwargs):
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
            logger (Logger): The logger for the timer. Default to None.
            batch_size (int): The batch size. Default to 1.
        """
        assert warmup >= 1
        if logger is None:
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
            cls.names[func_name]['batch_size'] = batch_size
            cls.names[func_name]['enable'] = True
        else:
            for name in cls.names:
                cls.names[name]['warmup'] = warmup
                cls.names[name]['log_interval'] = log_interval
                cls.names[name]['with_sync'] = with_sync
                cls.names[name]['batch_size'] = batch_size
                cls.names[name]['enable'] = True
        yield
        if func_name is not None:
            cls.names[func_name]['enable'] = False
        else:
            for name in cls.names:
                cls.names[name]['enable'] = False

    @classmethod
    def print_stats(cls, name: str):
        """print statistics results of timer.

        Args:
            name (str): The name registered with `count_time`.
        """
        from prettytable import PrettyTable

        assert name in cls.names
        stats = cls.names[name]
        execute_time = stats['execute_time']
        latency_mean = 1000 * float(np.mean(execute_time))
        latency_median = 1000 * float(np.median(execute_time))
        latency_min = 1000 * float(np.min(execute_time))
        latency_max = 1000 * float(np.max(execute_time))
        fps_mean, fps_median = 1000 / latency_mean, 1000 / latency_median
        fps_min, fps_max = 1000 / latency_min, 1000 / latency_max
        results = PrettyTable()
        results.field_names = ['Stats', 'Latency/ms', 'FPS']
        results.add_rows([
            ['Mean', latency_mean, fps_mean],
            ['Median', latency_median, fps_median],
            ['Min', latency_min, fps_min],
            ['Max', latency_max, fps_max],
        ])
        results.float_format = '.3'
        print(results)

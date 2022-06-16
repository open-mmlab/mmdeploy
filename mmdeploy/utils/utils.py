# Copyright (c) OpenMMLab. All rights reserved.
import glob
import logging
import os
import sys
import traceback
from typing import Callable, Optional, Union

try:
    from torch import multiprocessing as mp
except ImportError:
    import multiprocess as mp

from mmdeploy.utils.logging import get_logger


def target_wrapper(target: Callable,
                   log_level: int,
                   ret_value: Optional[mp.Value] = None,
                   *args,
                   **kwargs):
    """The wrapper used to start a new subprocess.

    Args:
        target (Callable): The target function to be wrapped.
        log_level (int): Log level for logging.
        ret_value (mp.Value): The success flag of target.

    Return:
        Any: The return of target.
    """
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger.level
    logger.setLevel(log_level)
    if ret_value is not None:
        ret_value.value = -1
    try:
        result = target(*args, **kwargs)
        if ret_value is not None:
            ret_value.value = 0
        return result
    except Exception as e:
        logging.error(e)
        traceback.print_exc(file=sys.stdout)


def get_root_logger(log_file=None, log_level=logging.INFO) -> logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        logging.Logger: The obtained logger
    """
    logger = get_logger(
        name='mmdeploy', log_file=log_file, log_level=log_level)

    return logger


def deprecate(status: str = 'future',
              dst_obj: Optional[Union[object, str]] = None,
              msg: str = '',
              *args,
              **kwargs) -> None:
    """Deprecate a function or a class.

    Args:
        status (str, optional): The status of the function or class.
            Defaults to future.
        dst_obj (str, object, optional): The object that will replace
            the original one. Defaults to None.
        msg (str): Additional message to be printed.

    Examples:
        >>> from math import ceil
        >>> from mmdeploy.utils.utils import deprecate
        >>> @deprecate(status='past', dst_obj=ceil, msg='')
        >>> def my_ceil(num):
        >>>     num = num if(num==int(num)) else int(num) + 1
        >>>     return num
    """
    logger = get_root_logger()

    def _register(src_obj):

        def fun(*args, **kwargs):
            if status == 'future':
                logger.warning(
                    f'DeprecationWarning: {src_obj.__name__} will be '
                    f'deprecated in the future. {msg}')
            elif status == 'past':
                assert dst_obj is not None, 'for deprecated object, there'
                ' must be a destination object'
                logger.warning(
                    f'DeprecationWarning: {src_obj.__name__} was deprecated,'
                    f' use {dst_obj.__name__} instead. {msg}')
            else:
                raise KeyError(f'Unexpected key {status}')
            result = src_obj(*args, **kwargs)
            return result

        return fun

    return _register


def get_file_path(prefix, candidates) -> str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        candidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ''

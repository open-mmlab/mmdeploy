import logging
import sys
import traceback
from typing import Callable, Optional

import torch.multiprocessing as mp


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
        format='%(asctime)s,%(name)s %(levelname)-8s'
        ' [%(filename)s:%(lineno)d] %(message)s',
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

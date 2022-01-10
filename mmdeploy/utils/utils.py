# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger


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

# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Optional


def parse_device_id(device: str) -> Optional[int]:
    """Parse device index from a string.

    Args:
        device (str): The typical style of string specifying device,
            e.g.: 'cuda:0', 'cpu'.

    Returns:
        Optional[int]: The return value depends on the type of device.
            If device is 'cuda': cuda device index, defaults to `0`.
            If device is 'cpu': `-1`.
            Otherwise, `None` will be returned.
    """
    if device == 'cpu':
        return -1
    if 'cuda' in device:
        return parse_cuda_device_id(device)
    return None


def parse_cuda_device_id(device: str) -> int:
    """Parse cuda device index from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0'.

    Returns:
        int: The parsed device id, defaults to `0`.
    """
    match_result = re.match('([^:]+)(:[0-9]+)?$', device)
    assert match_result is not None, f'Can not parse device {device}.'
    assert match_result.group(1).lower() == 'cuda', 'Not cuda device.'

    device_id = 0 if match_result.lastindex == 1 else int(
        match_result.group(2)[1:])

    return device_id


def parse_device_type(device: str) -> str:
    """Parse device type from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0', 'cpu', 'npu'.

    Returns:
        str: The parsed device type such as 'cuda', 'cpu', 'npu'.
    """
    device_type = device
    if ':' in device:
        device_type = device.split(':')[0]
    return device_type

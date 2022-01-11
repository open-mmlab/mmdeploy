# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch


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
    device = torch.device(device)
    assert device.type == 'cuda', 'Not cuda device.'

    device_id = 0 if device.index is None else device.index

    return device_id

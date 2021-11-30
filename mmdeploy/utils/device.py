# Copyright (c) OpenMMLab. All rights reserved.
import torch


def parse_device_id(device: str) -> int:
    """Parse cuda device index from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0'.

    Returns:
        int: The parsed device id, defaults to `0`.
    """
    if device == 'cpu':
        return -1
    device_id = 0
    if len(device) >= 6:
        device_id = torch.device(device).index
    return device_id


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

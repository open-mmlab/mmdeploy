from typing import List, Union

import numpy as np
import torch
from torch import nn


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def assert_allclose(actual: List[Union[torch.Tensor, np.ndarray]],
                    expected: List[Union[torch.Tensor, np.ndarray]],
                    tolerate_small_mismatch: bool = False):
    """Determine whether all actual values are closed with the expected values.

    Args:
        actual (list[torch.Tensor | np.ndarray): Actual value.
        expected (list[torch.Tensor | np.ndarray): Expected value.
        tolerate_small_mismatch (bool): Whether tolerate small mismatch,
        Default is False.
    """
    if not (isinstance(expected, list) and isinstance(actual, list)):
        raise ValueError('Argument desired and actual should be a list')
    if len(expected) != len(actual):
        raise ValueError('Length of desired and actual should be equal')

    for i in range(0, len(expected)):
        if isinstance(expected[i], (list, np.ndarray)):
            expected[i] = torch.tensor(expected[i])
        if isinstance(actual[i], (list, np.ndarray)):
            actual[i] = torch.tensor(actual[i])
        try:
            torch.testing.assert_allclose(
                actual[i], expected[i], rtol=1e-03, atol=1e-05)
        except AssertionError as error:
            if tolerate_small_mismatch:
                assert '(0.00%)' in str(error), str(error)
            else:
                raise

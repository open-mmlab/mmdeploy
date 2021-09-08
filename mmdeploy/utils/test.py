import numpy as np
import torch
from torch import nn


class WrapFunction(nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def assert_allclose(desired, actual, tolerate_small_mismatch=False):
    if not (isinstance(desired, list) and isinstance(actual, list)):
        raise ValueError('Argument desired and actual should be a list')
    if len(desired) != len(actual):
        raise ValueError('Length of desired and actual should be equal')

    for i in range(0, len(desired)):
        if isinstance(desired[i], (list, np.ndarray)):
            desired[i] = torch.tensor(desired[i])
        if isinstance(actual[i], (list, np.ndarray)):
            actual[i] = torch.tensor(actual[i])
        try:
            torch.testing.assert_allclose(
                desired[i], actual[i], rtol=1e-03, atol=1e-05)
        except AssertionError as error:
            if tolerate_small_mismatch:
                assert '(0.00%)' in str(error), str(error)
            else:
                raise

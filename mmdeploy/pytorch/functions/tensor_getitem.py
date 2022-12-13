# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.__getitem__', backend='ascend')
def tensor__getitem__ascend(self, key) -> torch.Tensor:
    """Rewrite `getitem` for ascend backend.

    Ascend does not support negative select
    """
    ctx = FUNCTION_REWRITER.get_context()
    if not isinstance(key, (tuple, list)):
        if isinstance(key, int) and key < 0:
            key = self.dim() + key
        return ctx.origin_func(self, key)

    def _num_slice_types(slices):
        num_slice = 0
        for s in slices:
            if isinstance(s, slice) or isinstance(s, int) or isinstance(
                    s, Iterable):
                num_slice += 1
        return num_slice

    shape = self.shape
    new_key = list(key)
    num_ellipsis = len(shape) - _num_slice_types(new_key)
    dim_count = 0
    for i, k in enumerate(new_key):
        if isinstance(k, int):
            if k < 0:
                new_key[i] = shape[dim_count] + k
        if k == Ellipsis:
            dim_count = dim_count + num_ellipsis
        elif k is not None:
            dim_count += 1
    return ctx.origin_func(self, new_key)

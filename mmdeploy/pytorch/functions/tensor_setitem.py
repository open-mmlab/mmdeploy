# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from packaging.version import parse

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.Tensor.__setitem__')
def tensor__setitem__default(ctx, self, key, value):
    """Rewrite `setitem` to ease the index put."""

    # only support torch>=1.9.0
    if parse(torch.__version__) < parse('1.9.0'):
        return ctx.origin_func(self, key, value)

    if isinstance(key, slice):
        key = (key, )

    if not isinstance(key, Sequence):
        return ctx.origin_func(self, key, value)

    for k in key:
        if not isinstance(k, slice) or k.step is not None:
            return ctx.origin_func(self, key, value)

    out = value
    for i, k in enumerate(key):
        if k == slice(None):
            continue

        cat_list = []

        # slice self start
        if k.start is not None:
            self_slice_start = (slice(None), ) * i + (slice(
                0, k.start), ) + key[i + 1:]
            self_start = self[self_slice_start]
            cat_list.append(self_start)

        # add value
        cat_list.append(out)

        # slice self end
        if k.stop is not None:
            self_slice_end = (slice(None), ) * i + (slice(
                k.stop, None), ) + key[i + 1:]
            self_end = self[self_slice_end]
            cat_list.append(self_end)

        # concate
        out = torch.cat(cat_list, dim=i)

    # self assign
    # Note that set item does not return any value
    self[...] = out

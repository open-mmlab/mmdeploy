# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import IR


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.chunk', backend='ncnn')
def chunk__ncnn(ctx, self, num_chunks: int, dim: int = 0) -> torch.Tensor:
    """Rewrite `chunk` for NCNN backend.

    Chunk in ncnn are not supported, so it should be rewritten.
    """
    dim_len = self.shape[dim]
    # int ceil.
    step = dim_len // num_chunks
    if dim_len % num_chunks > 0:
        step += 1
    index_list = []
    index = 0
    while index < dim_len:
        index_list.append(index)
        index += step
    index_list.append(dim_len)
    output = [
        self.index_select(
            dim,
            torch.tensor([j for j in range(index_list[i], index_list[i + 1])],
                         dtype=torch.int64))
        for i in range(len(index_list) - 1)
    ]

    return output


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.chunk', ir=IR.TORCHSCRIPT)
def chunk__torchscript(ctx,
                       self,
                       num_chunks: int,
                       dim: int = 0) -> torch.Tensor:
    """Rewrite `chunk` for Torchscript.

    Replace chunk op with split op
    """
    dim_size = self.shape[dim]
    assert dim_size % num_chunks == 0, 'cannot split to equal sizes'
    output = self.split(dim_size // num_chunks, dim=dim)
    return output

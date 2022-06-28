# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.onnx.symbolic_helper as sym_help

from mmdeploy.core import FUNCTION_REWRITER


# modified from
# https://github.com/pytorch/pytorch/blob/65a37923f9b14c7c9e80535d771ef9e4e92d0502/torch/onnx/symbolic_opset11.py
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.onnx.symbolic_opset11._prepare_onnx_paddings',
    backend='tensorrt')
def _prepare_onnx_paddings__tensorrt(ctx, g, input, pad):
    """Rewrite `_prepare_onnx_paddings` for TensorRT backend.

    For codes like `x = torch.nn.ZeroPad2d((0, a, 0, b))(x)`, where a and b are
    variables of torch.tensor, onnx2tensorrt raises errors like
    `INVALID_NODE: Invalid Node - Pad_`.

    Generate paddings in ONNX order based on pad in pytorch.
    Args:
        input: the input tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end,
            ..., dim_m_begin, dim_m_end,
            where m is in range [0, n].
    """
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # Assume zero-dimensions in the beginning, pad the "pad" sequence with
    # zeros in the beginning
    pad_len = torch.onnx.symbolic_opset9.size(
        g, pad, g.op('Constant', value_t=torch.tensor([0])))
    # Set extension = [0] * (dim * 2 - len(pad))
    rank = sym_help._get_tensor_rank(input)
    if rank is None:
        rank = g.op('Size', g.op('Shape', input))
    else:
        rank = g.op('Constant', value_t=torch.tensor(rank, dtype=torch.int64))
    extension = g.op(
        'Sub',
        g.op('Mul', rank,
             g.op('Constant', value_t=torch.tensor(2, dtype=torch.int64))),
        pad_len)
    # Concat pad with extension: paddings = [dim_n_begin, dim_n_end,
    # dim_n-1_begin, dim_n-1_end, 0, 0, ... ]
    # Currently ONNX only supports int64 type for Pad
    pad = g.op('Cast', pad, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    paddings = g.op(
        'Concat',
        pad,
        g.op(
            'ConstantOfShape',
            extension,
            value_t=torch.tensor([0], dtype=torch.int64)),
        axis_i=0)
    # Reshape and reverse order and collate first beginnings and then ends
    # paddings = [[..., 0, dim_n-1_begin, dim_n_begin],
    #               [..., 0, dim_n-1_end, dim_n_end]]
    # Reshape back to 1-D paddings = [..., 0, dim_n - 1_begin, dim_n_begin,
    # ..., 0, dim_n - 1_end, dim_n_end]

    # replace original Constant-Transpose-Constant with Slices and Concat.
    paddings = torch.onnx.symbolic_opset10.flip(g, paddings, [0])
    begins = sym_help._slice_helper(
        g, paddings, axes=[0], starts=[1], ends=[0xffff], steps=[2])
    ends = sym_help._slice_helper(
        g, paddings, axes=[0], starts=[0], ends=[0xffff], steps=[2])
    paddings = g.op('Concat', begins, ends, axis_i=0)
    padding_c = g.op(
        'Cast', paddings, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    return padding_c

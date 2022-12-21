# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.onnx.symbolic_helper as sym_help

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import get_ir_config


@SYMBOLIC_REWRITER.register_symbolic('squeeze', is_pytorch=True)
def squeeze__default(g, self, dim=None):
    """Register default symbolic function for `squeeze`.

    squeeze might be exported with IF node in ONNX, which is not supported in
    lots of backend.
    """
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [sym_help._get_const(dim, 'i', 'dim')]

    ctx = SYMBOLIC_REWRITER.get_context('squeeze')
    if get_ir_config(ctx.cfg).get('opset_version', 11) >= 13:
        axes = g.op('Constant', value_t=torch.tensor(dims, dtype=torch.long))
        return g.op('Squeeze', self, axes)

    return g.op('Squeeze', self, axes_i=dims)

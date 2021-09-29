import torch.onnx.symbolic_helper as sym_help

from mmdeploy.core import SYMBOLIC_REGISTER


@SYMBOLIC_REGISTER.register_symbolic('squeeze', is_pytorch=True)
def squeeze_default(ctx, g, self, dim=None):
    """Register default symbolic function for `squeeze`."""
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [sym_help._get_const(dim, 'i', 'dim')]
    return g.op('Squeeze', self, axes_i=dims)

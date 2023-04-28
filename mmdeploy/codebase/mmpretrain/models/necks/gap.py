# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    'mmpretrain.models.necks.GlobalAveragePooling.forward',
    backend=Backend.DEFAULT.value)
def gap__forward(self, inputs):
    """Rewrite `forward` of GlobalAveragePooling for default backend.

    Replace `view` with `flatten` to export simple onnx graph.
    Shape->Gather->Unsqueeze->Concat->Reshape become a Flatten.
    """
    if isinstance(inputs, tuple):
        outs = tuple([self.gap(x) for x in inputs])
        outs = tuple([out.flatten(1) for out in outs])
    elif isinstance(inputs, torch.Tensor):
        outs = self.gap(inputs)
        outs = outs.flatten(1)
    else:
        raise TypeError('neck inputs should be tuple or torch.tensor')
    return outs

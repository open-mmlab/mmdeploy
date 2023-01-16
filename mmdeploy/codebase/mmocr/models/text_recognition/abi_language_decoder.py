# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import IR


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.ABILanguageDecoder._get_length',
    IR=IR.ONNX)
def abi_language_decoder___get_length__default(self,
                                               logit: torch.Tensor,
                                               dim: int = -1,
                                               **kwargs) -> torch.Tensor:
    """Rewrite `_get_length`. Add `.float()` to cast Tensors from bool to float
    for `cumsum` and `argmax`.

    Returns the first location of padding index or the length of the entire
    tensor otherwise.
    """
    # out as a boolean vector indicating the existence of end token(s)
    out = (logit.argmax(dim=-1) == self.dictionary.end_idx)
    abn = out.any(dim)
    # Get the first index of end token
    # add `.float()` to `out` for onnxruntime `cumsum()`
    # add `.float()` before `argmax()`
    out = ((out.float().cumsum(dim) == 1) & out).float().argmax(dim)
    out = out + 1
    out = torch.where(abn, out,
                      out.new_tensor(logit.shape[1]).to(out.device)).float()
    return out

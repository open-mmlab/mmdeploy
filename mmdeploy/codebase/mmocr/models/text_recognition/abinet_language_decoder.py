# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.ABILanguageDecoder'
    '._get_length',
    backend='default')
def abi_language_decoder__get_length(ctx, self, logit, dim=-1):
    """Rewrite `_get_length` of ABILanguageDecoder for default backend. Greedy
    decoder to obtain length from logit. Returns the first location of padding
    index or the length of the entire tensor otherwise.

    Rewrite this function to:
    1. use torch.max to replace original torch.any and torch.cumsum.
    2. modify the data type of out.new_tensor().
    """
    # out as a boolean vector indicating the existence of end token(s)
    out = (logit.argmax(dim=-1) == self.pad_idx)
    out = out.to(torch.float32)
    abn, out = torch.max(out, dim)
    abn = abn > 0
    # Get the first index of end token
    out = out + 1
    out = torch.where(abn, out,
                      out.new_tensor(logit.shape[1], device=out.device))
    return out


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.ABILanguageDecoder'
    '.forward_train')
def abi_language_decoder__forward_train(ctx, self, feat, logits, targets_dict,
                                        img_metas):
    """Rewrite `forward_train` of ABILanguageDecoder for default backend.

    Rewrite this function to support nn.MultiheadAttention in backend
    when attn_masks is not None.

    Args:
            logits (Tensor): Raw language logitis. Shape (N, T, C).

        Returns:
            A dict with keys ``feature`` and ``logits``.
            feature (Tensor): Shape (N, T, E). Raw textual features for vision
                language aligner.
            logits (Tensor): Shape (N, T, C). The raw logits for characters
                after spell correction.
    """
    lengths = self._get_length(logits)
    lengths.clamp_(2, self.max_seq_len)
    tokens = torch.softmax(logits, dim=-1)
    if self.detach_tokens:
        tokens = tokens.detach()
    embed = self.proj(tokens)  # (N, T, E)
    embed = self.token_encoder(embed)  # (N, T, E)
    padding_mask = self._get_padding_mask(lengths, self.max_seq_len)

    zeros = embed.new_zeros(*embed.shape)
    query = self.pos_encoder(zeros)
    query = query.permute(1, 0, 2)  # (T, N, E)
    embed = embed.permute(1, 0, 2)
    output = query
    for m in self.decoder_layers:
        output = m(
            query=output,
            key=embed,
            value=embed,
            key_padding_mask=padding_mask)
    output = output.permute(1, 0, 2)  # (N, T, E)

    logits = self.cls(output)  # (N, T, C)
    return {'feature': output, 'logits': logits}

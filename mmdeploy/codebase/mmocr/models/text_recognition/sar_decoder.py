# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmocr.utils.typing_utils import TextRecogDataSample
from torch import nn

from mmdeploy.core import FUNCTION_REWRITER, MODULE_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.ParallelSARDecoder'
    '._2d_attention',
    backend='default')
def parallel_sar_decoder__2d_attention(
        self,
        decoder_input: torch.Tensor,
        feat: torch.Tensor,
        holistic_feat: torch.Tensor,
        valid_ratios: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Rewrite `_2d_attention` of ParallelSARDecoder for default backend.

    Rewrite this function to:
    1. use torch.ceil to replace original math.ceil and if else in mmocr.
    2. use narrow to replace original [valid_width:] in mmocr
    """
    y = self.rnn_decoder(decoder_input)[0]
    # y: bsz * (seq_len + 1) * hidden_size

    attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
    bsz, seq_len, attn_size = attn_query.size()
    attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

    attn_key = self.conv3x3_1(feat)
    # bsz * attn_size * h * w
    attn_key = attn_key.unsqueeze(1)
    # bsz * 1 * attn_size * h * w

    attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
    # bsz * (seq_len + 1) * attn_size * h * w
    attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
    # bsz * (seq_len + 1) * h * w * attn_size
    attn_weight = self.conv1x1_2(attn_weight)
    # bsz * (seq_len + 1) * h * w * 1
    bsz, T, h, w, c = attn_weight.size()
    assert c == 1

    if valid_ratios is not None:
        # cal mask of attention weight
        attn_mask = torch.zeros(bsz, T, h, w + 1, c).to(attn_weight.device)
        for i, valid_ratio in enumerate(valid_ratios):
            # use torch.ceil to replace original math.ceil and if else in mmocr
            valid_width = torch.tensor(w * valid_ratio).ceil().long()
            # use narrow to replace original [valid_width:] in mmocr
            attn_mask[i].narrow(2, valid_width, w + 1 - valid_width)[:] = 1
        attn_mask = attn_mask[:, :, :, :w, :]
        attn_weight = attn_weight.masked_fill(attn_mask.bool(), float('-inf'))

    attn_weight = attn_weight.view(bsz, T, -1)
    attn_weight = F.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.view(bsz, T, h, w, c).permute(0, 1, 4, 2,
                                                            3).contiguous()

    attn_feat = torch.sum(
        torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
    # bsz * (seq_len + 1) * C

    # linear transformation
    if self.pred_concat:
        hf_c = holistic_feat.size(-1)
        holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
        y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
    else:
        y = self.prediction(attn_feat)
    # bsz * (seq_len + 1) * num_classes
    y = self.pred_dropout(y)

    return y


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.SequentialSARDecoder'
    '._2d_attention',
    backend='default')
def sequential_sar_decoder__2d_attention(self,
                                         y_prev,
                                         feat,
                                         holistic_feat,
                                         hx1,
                                         cx1,
                                         hx2,
                                         cx2,
                                         valid_ratios=None):
    """Rewrite `_2d_attention` of SequentialSARDecoder for default backend.

    Rewrite this function to:
    1. use torch.ceil to replace original math.ceil and if else in mmocr.
    2. use narrow to replace original [valid_width:] in mmocr
    """
    _, _, h_feat, w_feat = feat.size()
    if self.dec_gru:
        hx1 = cx1 = self.rnn_decoder_layer1(y_prev, hx1)
        hx2 = cx2 = self.rnn_decoder_layer2(hx1, hx2)
    else:
        # has replaced LSTMCell with LSTM, forward func need rewrite
        _, (hx1,
            cx1) = self.rnn_decoder_layer1(y_prev.unsqueeze(0), (hx1, cx1))
        _, (hx2, cx2) = self.rnn_decoder_layer2(hx1, (hx2, cx2))

    tile_hx2 = hx2.view(hx2.size(1), hx2.size(-1), 1, 1)
    attn_query = self.conv1x1_1(tile_hx2)  # bsz * attn_size * 1 * 1
    attn_query = attn_query.expand(-1, -1, h_feat, w_feat)
    attn_key = self.conv3x3_1(feat)
    attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
    attn_weight = self.conv1x1_2(attn_weight)
    bsz, c, h, w = attn_weight.size()
    assert c == 1

    if valid_ratios is not None:
        # cal mask of attention weight
        attn_mask = torch.zeros(bsz, c, h, w + 1).to(attn_weight.device)
        for i, valid_ratio in enumerate(valid_ratios):
            # use torch.ceil to replace original math.ceil and if else in mmocr
            valid_width = torch.tensor(w * valid_ratio).ceil().long()
            # use narrow to replace original [valid_width:] in mmocr
            attn_mask[i].narrow(2, valid_width, w + 1 - valid_width)[:] = 1
        attn_mask = attn_mask[:, :, :, :w]
        attn_weight = attn_weight.masked_fill(attn_mask.bool(), float('-inf'))

    attn_weight = F.softmax(attn_weight.view(bsz, -1), dim=-1)
    attn_weight = attn_weight.view(bsz, c, h, w)

    attn_feat = torch.sum(
        torch.mul(feat, attn_weight), (2, 3), keepdim=False)  # n * c

    # linear transformation
    if self.pred_concat:
        y = self.prediction(torch.cat((hx2[0], attn_feat, holistic_feat), 1))
    else:
        y = self.prediction(attn_feat)

    return y, hx1, hx1, hx2, hx2


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.SequentialSARDecoder'
    '.forward_test',
    backend='default')
def sequential_sar_decoder__forward_test(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None):
    """Rewrite `forward_test` of SequentialSARDecoder for default backend.

    Rewrite this function because LSTMCell has been replaced with LSTM. The two
    class have different forward functions. The `forward_test` need adapt to
    this change.
    """
    valid_ratios = None
    if data_samples is not None:
        valid_ratios = [
            data_sample.get('valid_ratio', 1.0) for data_sample in data_samples
        ] if self.mask else None

    outputs = []
    start_token = torch.full((feat.size(0), ),
                             self.start_idx,
                             device=feat.device,
                             dtype=torch.long)
    start_token = self.embedding(start_token)
    for i in range(-1, self.max_seq_len):
        if i == -1:
            if self.dec_gru:
                hx1 = cx1 = self.rnn_decoder_layer1(out_enc)
                hx2 = cx2 = self.rnn_decoder_layer2(hx1)
            else:
                # has replaced LSTMCell with LSTM, forward func need rewrite
                _, (hx1, cx1) = self.rnn_decoder_layer1(out_enc.unsqueeze(0))
                _, (hx2, cx2) = self.rnn_decoder_layer2(hx1)
                y_prev = start_token
        else:
            y, hx1, cx1, hx2, cx2 = self._2d_attention(
                y_prev,
                feat,
                out_enc,
                hx1,
                cx1,
                hx2,
                cx2,
                valid_ratios=valid_ratios)
            _, max_idx = torch.max(y, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)
            y_prev = char_embedding
            outputs.append(y)

    outputs = torch.stack(outputs, 1)

    return outputs


@MODULE_REWRITER.register_rewrite_module(
    'mmocr.models.textrecog.decoders.SequentialSARDecoder', backend='default')
class SequentialSARDecoder(nn.Module):
    """Rewrite Sequential Decoder module in `SAR.

    SequentialSARDecoder apply nn.LSTMCell inside, which brings obstacles to
    deployment. LSTMCell can be only exported to onnx in cpu device. To make it
    exportable to gpu device, LSTM is used to replace LSTMCell.

    <https://arxiv.org/abs/1811.00751>`_.
    """

    def __init__(self, module, deploy_cfg, **kwargs):
        super(SequentialSARDecoder, self).__init__()

        def lstmcell2lstm_params(lstm_mod, lstmcell_mod):
            lstm_mod.weight_ih_l0 = nn.Parameter(lstmcell_mod.weight_ih)
            lstm_mod.weight_hh_l0 = nn.Parameter(lstmcell_mod.weight_hh)
            lstm_mod.bias_ih_l0 = nn.Parameter(lstmcell_mod.bias_ih)
            lstm_mod.bias_hh_l0 = nn.Parameter(lstmcell_mod.bias_hh)

        self._module = module
        self.deploy_cfg = deploy_cfg
        if not self._module.dec_gru:
            rnn_decoder_layer1 = copy.deepcopy(self._module.rnn_decoder_layer1)
            rnn_decoder_layer2 = copy.deepcopy(self._module.rnn_decoder_layer2)
            self._module.rnn_decoder_layer1 = nn.LSTM(
                rnn_decoder_layer1.input_size, rnn_decoder_layer1.hidden_size,
                1)
            self._module.rnn_decoder_layer2 = nn.LSTM(
                rnn_decoder_layer2.input_size, rnn_decoder_layer2.hidden_size,
                1)
            lstmcell2lstm_params(self._module.rnn_decoder_layer1,
                                 rnn_decoder_layer1)
            lstmcell2lstm_params(self._module.rnn_decoder_layer2,
                                 rnn_decoder_layer2)
        self._module.train_mode = False

    def forward(self,
                feat: Optional[torch.Tensor] = None,
                out_enc: Optional[torch.Tensor] = None,
                data_samples: Optional[Sequence[TextRecogDataSample]] = None):
        return self._module.forward_test(feat, out_enc, data_samples)

    def predict(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> Sequence[TextRecogDataSample]:
        """Perform forward propagation of the decoder and postprocessor.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder. Defaults
                to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        out_dec = self(feat, out_enc, data_samples)
        return out_dec

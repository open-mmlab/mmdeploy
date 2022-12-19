# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.layers.lstm_layer'
    '.BidirectionalLSTM.forward',
    backend='ncnn')
def bidirectionallstm__forward__ncnn(self, input):
    """Rewrite `forward` of BidirectionalLSTM for ncnn backend.

    Rewrite this function to set batch_first of rnn layer to true. RNN in ncnn
    requires batch first.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class
            BidirectionalLSTM.
        input (Tensor): Input tensor of shape (N, H, W).

    Returns:
        output (Tensor): Embedded tensor from embedding layer.
    """

    self.rnn.batch_first = True
    recurrent, _ = self.rnn(input)
    self.rnn.batch_first = False

    output = self.embedding(recurrent)

    return output

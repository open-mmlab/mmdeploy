from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.layers.lstm_layer'
    '.BidirectionalLSTM.forward',
    backend='ncnn')
def forward_of_bidirectionallstm(ctx, self, input):
    self.rnn.batch_first = True
    recurrent, _ = self.rnn(input)
    self.rnn.batch_first = False

    output = self.embedding(recurrent)

    return output

# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.cnn import MultiHeadAttentionop
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.utils.attention.MultiheadAttention.forward',
    backend=Backend.NCNN.value)
def multiheadattention__forward__ncnn(ctx, self, qkv_input):
    """Rewrite `forward` of MultiheadAttention used in vision_transformer for
    ncnn backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (MultiheadAttention): The instance of the class
        MultiheadAttention.
        x (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from MultiHeadAttention. The tensor
        shape (N, Cout, H, W).
    """

    # split qkv weight and bias
    qkv_weight = self.qkv.weight.data.reshape(3, self.input_dims,
                                              self.embed_dims)

    q_weight = qkv_weight[0]
    k_weight = qkv_weight[1]
    v_weight = qkv_weight[2]

    qkv_bias = self.qkv.bias.data.reshape(3, self.embed_dims)
    q_bias = qkv_bias[0]
    k_bias = qkv_bias[1]
    v_bias = qkv_bias[2]

    # out weight and bias
    o_weight = self.proj.weight.data
    o_bias = self.proj.bias.data

    out = MultiHeadAttentionop.apply(qkv_input, qkv_input, qkv_input, q_weight,
                                     q_bias, k_weight, k_bias, v_weight,
                                     v_bias, o_weight, o_bias, self.embed_dims,
                                     self.num_heads)
    return out

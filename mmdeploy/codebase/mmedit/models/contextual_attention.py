# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmedit.models.common.contextual_attention.ContextualAttentionModule.'
    'patch_correlation')
def contextual_attention__patch_correlation(ctx, self, x, kernel):
    # Force tensor shape to avoid the following RuntimeError:
    # Unsupported: ONNX export of convolution for kernel of unknown shape.
    kernel_shape = ctx.cfg.codebase_config.get('kernel_shape',
                                               (1024, 96, 3, 3))
    kernel += kernel.new_zeros(kernel_shape)
    return ctx.origin_func(self, x, kernel)


@FUNCTION_REWRITER.register_rewriter(
    'mmedit.models.common.contextual_attention.ContextualAttentionModule.'
    'patch_copy_deconv')
def contextual_attention__patch_copy_deconv(ctx, self, attention_score,
                                            context_filter):
    # Force tensor shape to avoid the following RuntimeError:
    # Unsupported: ONNX export of convolution for kernel of unknown shape.
    context_filter_shape = ctx.cfg.codebase_config.get('context_filter_shape',
                                                       (1024, 96, 4, 4))
    context_filter += context_filter.new_zeros(context_filter_shape)
    return ctx.origin_func(self, attention_score, context_filter)

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.nn.SyncBatchNorm.forward'
                                     )
def sync_batch_norm(ctx, self, input: torch.Tensor) -> torch.Tensor:
    """Rewrite `SyncBatchNorm` for CPU backends.
    """
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum
    bn_training = (self.running_mean is None) and (self.running_var is None)
    # If buffers are not to be tracked, ensure that they won't be updated
    assert self.running_mean is None or isinstance(self.running_mean,
                                                   torch.Tensor)
    assert self.running_var is None or isinstance(self.running_var,
                                                  torch.Tensor)
    running_mean = self.running_mean if not self.training or\
        self.track_running_stats else None
    running_var = self.running_var if not self.training or\
        self.track_running_stats else None

    # fallback to framework BN when synchronization is not necessary
    return F.batch_norm(input, running_mean, running_var, self.weight,
                        self.bias, bn_training, exponential_average_factor,
                        self.eps)

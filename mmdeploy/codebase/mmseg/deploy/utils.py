# Copyright (c) OpenMMLab. All rights reserved.
import torch


def convert_syncbatchnorm(module: torch.nn.Module):
    """Convert sync batch-norm to batch-norm for inference.

    Args:
        module (nn.Module): Input PyTorch model.

    Returns:
        nn.Module: PyTorch model without sync batch-norm.
    """
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_syncbatchnorm(child))
    del module
    return module_output

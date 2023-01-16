# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from mmdeploy.core import MODULE_REWRITER


@MODULE_REWRITER.register_rewrite_module(
    'mmocr.models.common.modules.PositionalEncoding', backend='default')
class PositionalEncoding(nn.Module):
    """Rewrite Position Encoding module in `ABINet."""

    def __init__(self, module, deploy_cfg, **kwargs):
        super(PositionalEncoding, self).__init__()
        self._module = module
        self.deploy_cfg = deploy_cfg
        self.n_position = module.position_table.size(1)
        self.d_hid = module.position_table.size(2)

    def _get_sinusoid_encoding_table(self, n_position, d_hid, device):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / torch.tensor(10000).to(device).pow(
                torch.tensor(2 * (hid_j // 2) / d_hid)).to(device)
            for hid_j in range(d_hid)
        ]).to(device)
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).to(device).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        device = x.device
        position_table = self._get_sinusoid_encoding_table(
            self.n_position, self.d_hid, device)
        x = x + position_table[:, :x.size(1), ...]
        return x

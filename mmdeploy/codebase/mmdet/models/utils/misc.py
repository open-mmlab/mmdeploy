import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.utils.misc.generate_coordinate')
def generate_coordinate__onnx(ctx, featmap_sizes, device="cuda"):
    x_range = torch.arange(-1, 2, 2. / (featmap_sizes[-1] - 1), device=device)[:featmap_sizes[-1]]
    y_range = torch.arange(-1, 2, 2. / (featmap_sizes[-2] - 1), device=device)[:featmap_sizes[-2]]
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([featmap_sizes[0], 1, -1, -1])
    x = x.expand([featmap_sizes[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)

    return coord_feat
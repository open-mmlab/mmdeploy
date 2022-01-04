import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.csp_darknet.Focus.forward',
    backend='ncnn')
def focus__forward__ncnn(ctx, self, x):
    """Rewrite forward function of Focus class for ncnn.

    Focus width and height information into channel space. NCNN does not
    support slice operator which step greater than 1, so we use another
    way to implement.

    Args:
        x (Tensor): The input tensor with shape (N, C, H, W).

    Returns:
        x (Tensor): The calculated tensor with shape (N, 4*C, H//2, W//2).
    """
    batch_size, c, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature\
        height and width, got {(h, w)}.'

    x_col = x.reshape(batch_size, c, -1, 2)
    left = x_col[:, :, :, 0:1]
    right = x_col[:, :, :, 1:2]
    left = left.reshape(batch_size, c, h, -1)
    left = left.permute(0, 1, 3, 2).reshape(batch_size, c, -1, 2)
    top_left = left[:, :, :, 0:1].reshape(batch_size, c, w//2, h//2).\
        permute(0, 1, 3, 2)
    bot_left = left[:, :, :, 1:2].reshape(batch_size, c, w//2, h//2).\
        permute(0, 1, 3, 2)
    right = right.reshape(batch_size, c, h, -1)
    right = right.permute(0, 1, 3, 2).reshape(batch_size, c, -1, 2)
    top_right = right[:, :, :, 0:1].reshape(batch_size, c, w//2, h//2).\
        permute(0, 1, 3, 2)
    bot_right = right[:, :, :, 1:2].reshape(batch_size, c, w//2, h//2).\
        permute(0, 1, 3, 2)

    x = torch.cat(
        (
            top_left,
            bot_left,
            top_right,
            bot_right,
        ),
        dim=1,
    )
    return self.conv(x)

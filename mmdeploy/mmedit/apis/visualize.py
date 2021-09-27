import warnings

import mmcv
import numpy as np
import torch

from mmdeploy.utils import Backend


# BaseModel in mmedit doesn't implement show_result
# TODO: add show_result to different tasks
def show_result(result: np.ndarray,
                output_file: str,
                backend: Backend,
                show: bool = True):
    """Show high resolution image of mmedit.

    Args:
        result: (np.ndarray): Input high resolution image.
        output_file (str): Output image file to save image.
        backend (Backend): Specifying backend type.
        show (bool): Whether to show plotted image in windows. Defaults to
            `True`.

    Returns:
        np.ndarray: Drawn image, only if not `show` or `out_file`.
    """
    win_name = backend.value
    with torch.no_grad():
        result = result.transpose(1, 2, 0)
        result = np.clip(result, 0, 1)[:, :, ::-1]
        result = (result * 255.0).round()

        if output_file is not None:
            show = False

        if show:
            int_result = result.astype(np.uint8)
            mmcv.imshow(int_result, win_name, 0)
        if output_file is not None:
            mmcv.imwrite(result, output_file)

    if not (show or output_file):
        warnings.warn('show==False and output_file is not specified, only '
                      'result image will be returned')
        return result

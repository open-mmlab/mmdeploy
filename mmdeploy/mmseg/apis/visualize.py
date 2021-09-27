import numpy as np
import torch

from mmdeploy.utils import Backend


def show_result(model: torch.nn.Module,
                image: np.ndarray,
                result: list,
                output_file: str,
                backend: Backend,
                show: bool = True,
                opacity: float = 0.5):
    """Show predictions of segmentation.

    Args:
        model (nn.Module): Input model which has `show_result` method.
        image: (np.ndarray): Input image to draw predictions.
        result (list): A list of predictions.
        output_file (str): Output image file to save drawn predictions.
        backend (Backend): Specifying backend type.
        show (bool): Whether to show plotted image in windows. Defaults to
            `True`.
        opacity: (float): Opacity of painted segmentation map.
                Defaults to `0.5`.

    Returns:
        np.ndarray: Drawn image, only if not `show` or `out_file`.
    """
    return model.show_result(
        image,
        result,
        opacity=opacity,
        show=show,
        win_name=backend.value,
        out_file=output_file)

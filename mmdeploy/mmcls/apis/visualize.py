import numpy as np
import torch

from mmdeploy.utils import Backend


def show_result(model: torch.nn.Module,
                image: np.ndarray,
                result: list,
                output_file: str,
                backend: Backend,
                show: bool = True):
    """Show predictions of mmcls.

    Args:
        model (nn.Module): Input model which has `show_result` method.
        image: (np.ndarray): Input image to draw predictions.
        result (list): A list of predictions.
        output_file (str): Output image file to save drawn predictions.
        backend (Backend): Specifying backend type.
        show (bool): Whether to show plotted image in windows. Defaults to
            `True`.

    Returns:
        np.ndarray: Drawn image, only if not `show` or `out_file`.
    """
    pred_score = np.max(result, axis=0)
    pred_label = np.argmax(result, axis=0)
    result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return model.show_result(
        image, result, show=show, win_name=backend.value, out_file=output_file)

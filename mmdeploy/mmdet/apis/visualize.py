import numpy as np

from mmdeploy.utils import Backend


def show_result(model,
                image: np.ndarray,
                result: list,
                output_file: str,
                backend: Backend,
                show: bool = True,
                score_thr: float = 0.3):
    """Show predictions of detection.

    Args:
        model (nn.Module): Input model which has `show_result` method.
        image: (np.ndarray): Input image to draw predictions.
        result (list): A list of predictions.
        output_file (str): Output image file to save drawn predictions.
        backend (Backend): Specifying backend type.
        show (bool): Whether to show plotted image in windows. Defaults to
            `True`.
        score_thr (float): Score threshold for detection, defaults to `0.3`.

    Returns:
        np.ndarray: Drawn image, only if not `show` or `out_file`.
    """
    return model.show_result(
        image,
        result,
        score_thr=score_thr,
        show=show,
        win_name=backend.value,
        out_file=output_file)

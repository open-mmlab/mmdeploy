import numpy as np

from mmdeploy.utils import Backend


def show_result(model,
                image: np.ndarray,
                result,
                output_file: str,
                backend: Backend,
                show=True,
                opacity=0.5):
    return model.show_result(
        image,
        result,
        opacity=opacity,
        show=show,
        win_name=backend.value,
        out_file=output_file)

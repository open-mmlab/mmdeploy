import numpy as np

from mmdeploy.utils import Backend


def show_result(model,
                image: np.ndarray,
                result,
                output_file: str,
                backend: Backend,
                show=True,
                score_thr=0.3):
    return model.show_result(
        image,
        result,
        score_thr=score_thr,
        show=show,
        win_name=backend.value,
        out_file=output_file)

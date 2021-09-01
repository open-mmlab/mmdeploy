import numpy as np

from mmdeploy.utils import Backend


def show_result(model,
                image: np.ndarray,
                result,
                output_file: str,
                backend: Backend,
                show=True):
    pred_score = np.max(result, axis=0)
    pred_label = np.argmax(result, axis=0)
    result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return model.show_result(
        image, result, show=show, win_name=backend.value, out_file=output_file)

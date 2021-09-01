import warnings

import mmcv
import numpy as np
import torch

from mmdeploy.utils import Backend


# BaseModel in mmedit doesn't implement show_result
# TODO: add show_result to different tasks
def show_result(result, output_file, backend: Backend, show=True):
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

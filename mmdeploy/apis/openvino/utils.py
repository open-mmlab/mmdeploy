from typing import List

import mmcv


def get_input_shape_from_cfg(config: mmcv.Config) -> List[int]:
    """Get the input shape from the model config for OpenVINO Model Optimizer.

    Args:
        config (mmcv.Config): Model config.
    Returns:
        List[int]: The input shape in [1, 3, H, W] format from config
            or [1, 3, 800, 1344].
    """
    shape = []
    test_pipeline = config.get('test_pipeline', None)
    if test_pipeline is not None:
        img_scale = test_pipeline[1]['img_scale']
        shape = [1, 3, img_scale[1], img_scale[0]]
    else:
        shape = [1, 3, 800, 1344]
    return shape

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mmcv

from mmdeploy.utils import get_input_shape


def get_input_shape_from_cfg(deploy_cfg: mmcv.Config,
                             model_cfg: mmcv.Config) -> List[int]:
    """Get the input shape from the configs for OpenVINO Model Optimizer. The
    value from config 'deploy_cfg' has the highest priority, then 'model_cfg'.
    If there is no input shape in configs, then the default value will be used.

    Args:
        deploy_cfg (mmcv.Config): Deployment config.
        model_cfg (mmcv.Config): Model config.
    Returns:
        List[int]: The input shape in [1, 3, H, W] format from config
            or [1, 3, 800, 1344].
    """
    shape = [1, 3]
    is_use_deploy_cfg = False
    try:
        input_shape = get_input_shape(deploy_cfg)
        if input_shape is not None:
            is_use_deploy_cfg = True
    except KeyError:
        is_use_deploy_cfg = False

    if is_use_deploy_cfg:
        shape += [input_shape[1], input_shape[0]]
    else:
        test_pipeline = model_cfg.get('test_pipeline', None)
        if test_pipeline is not None:
            img_scale = test_pipeline[1]['img_scale']
            shape += [img_scale[1], img_scale[0]]
        else:
            shape += [800, 1344]
    return shape

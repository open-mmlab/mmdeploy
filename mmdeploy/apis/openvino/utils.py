# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import mmcv

from mmdeploy.utils import get_model_inputs


def get_input_info_from_cfg(deploy_cfg: mmcv.Config) -> Dict[str, List]:
    """Get the input names and shapes from the configs for OpenVINO Model
    Optimizer.

    Args:
        deploy_cfg (mmcv.Config): Deployment config.

    Returns:
        Dict[str, List]: A dict that stores the names and shapes of input.
    """
    # The partition is not supported now. Set the id of model to 0.
    model_inputs = get_model_inputs(deploy_cfg)[0]
    input_info = model_inputs['opt_shapes']
    return input_info

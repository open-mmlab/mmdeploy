# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import mmcv

from mmdeploy.utils import load_config


def dump_info(deploy_cfg: Union[str, mmcv.Config],
              model_cfg: Union[str, mmcv.Config], work_dir: str):
    """Export information to SDK.

    Args:
        deploy_cfg (str | mmcv.Config): Deploy config file or dict.
        model_cfg (str | mmcv.Config): Model config file or dict.
        work_dir (str): Work dir to save json files.
    """
    # TODO dump default values of transformation function to json
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    meta_keys = [
        'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'
    ]
    if 'transforms' in model_cfg.data.test.pipeline[-1]:
        model_cfg.data.test.pipeline[-1]['transforms'][-1][
            'meta_keys'] = meta_keys
    else:
        model_cfg.data.test.pipeline[-1]['meta_keys'] = meta_keys
    mmcv.dump(
        model_cfg.data.test.pipeline,
        '{}/preprocess.json'.format(work_dir),
        sort_keys=False,
        indent=4)

    mmcv.dump(
        deploy_cfg._cfg_dict,
        '{}/deploy_cfg.json'.format(work_dir),
        sort_keys=False,
        indent=4)

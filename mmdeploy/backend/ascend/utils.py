# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp

from mmdeploy.utils import get_root_logger


def update_sdk_pipeline(work_dir: str):
    """Update pipeline.json for Ascend.

    Args:
        work_dir (str):The work directory to load/save the pipeline.json
    """
    logger = get_root_logger()

    def _try_ori_agnostic_pad(transforms):
        trans_resize = None
        trans_pad = None

        for trans in transforms:
            if trans['type'] == 'Resize' and trans.get('keep_ratio', False):
                trans_resize = trans
            elif trans['type'] == 'Pad' and trans.get('size_divisor',
                                                      None) is not None:
                trans_pad = trans

        if trans_resize is not None and trans_pad is not None:
            logger.info('update Pad transform.')
            size = trans_resize['size']
            divisor = trans_pad['size_divisor']
            size = tuple(int(math.ceil(s / divisor) * divisor) for s in size)
            trans_pad['size'] = size
            trans_pad['orientation_agnostic'] = True
            trans_pad.pop('size_divisor')

    pipeline_path = osp.join(work_dir, 'pipeline.json')

    if osp.exists(pipeline_path):
        import mmcv
        pipeline = mmcv.load(pipeline_path)
        tasks = pipeline['pipeline'].get('tasks', [])

        for task in tasks:
            if task.get('module', '') == 'Transform':
                transforms = task['transforms']
                _try_ori_agnostic_pad(transforms)

        mmcv.dump(pipeline, pipeline_path, sort_keys=False, indent=4)

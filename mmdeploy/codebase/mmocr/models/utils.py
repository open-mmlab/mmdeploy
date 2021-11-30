# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import mmcv

from mmdeploy.utils import load_config


def get_resize_ocr(model_cfg: Union[str, mmcv.Config]):
    """Get the test settings of ResizeOCR in model config.

    Args:
        model_cfg (str | mmcv.Config): Model config file or loaded Config
            object.

    Returns:
        tuple, composed of min_width, max_width and keep_aspect_ratio.
    """
    model_cfg = load_config(model_cfg)[0]
    from mmdet.datasets.pipelines import Compose
    from mmocr.datasets import build_dataset  # noqa: F401
    test_pipeline = Compose(model_cfg.data.test.pipeline)
    resize_ocr = test_pipeline.transforms[1].transforms.transforms[0]
    return (resize_ocr.min_width, resize_ocr.max_width,
            resize_ocr.keep_aspect_ratio)

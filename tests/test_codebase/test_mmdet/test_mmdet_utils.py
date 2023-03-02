# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase

try:
    import_codebase(Codebase.MMDET)
except ImportError:
    pytest.skip(f'{Codebase.MMDET} is not installed.', allow_module_level=True)

from mmdeploy.codebase.mmdet.deploy import (clip_bboxes,
                                            get_post_processing_params,
                                            pad_with_value,
                                            pad_with_value_if_necessary)


def test_clip_bboxes():
    x1 = torch.rand(3, 2) * 224
    y1 = torch.rand(3, 2) * 224
    x2 = x1 * 2
    y2 = y1 * 2
    outs = clip_bboxes(x1, y1, x2, y2, [224, 224])
    for out in outs:
        assert int(out.max()) <= 224


def test_pad_with_value():
    x = torch.rand(3, 2)
    padded_x = pad_with_value(x, pad_dim=1, pad_size=4, pad_value=0)
    assert np.allclose(
        padded_x.shape, torch.Size([3, 6]), rtol=1e-03, atol=1e-05)
    assert np.allclose(padded_x.sum(), x.sum(), rtol=1e-03, atol=1e-05)


def test_pad_with_value_if_necessary():
    x = torch.rand(3, 2)
    padded_x = pad_with_value_if_necessary(
        x, pad_dim=1, pad_size=4, pad_value=0)
    assert np.allclose(
        padded_x.shape, torch.Size([3, 2]), rtol=1e-03, atol=1e-05)
    assert np.allclose(padded_x.sum(), x.sum(), rtol=1e-03, atol=1e-05)


config_with_mmdet_params = Config(
    dict(
        codebase_config=dict(
            type='mmdet',
            task='ObjectDetection',
            post_processing=dict(
                score_threshold=0.05,
                iou_threshold=0.5,
                max_output_boxes_per_class=200,
                pre_top_k=-1,
                keep_top_k=100,
                background_label_id=-1,
            ))))


def test_get_mmdet_params():
    assert get_post_processing_params(config_with_mmdet_params) == dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=-1,
        keep_top_k=100,
        background_label_id=-1)


def test_get_topk_from_heatmap():
    from mmdet.models.utils.gaussian_target import get_topk_from_heatmap

    from mmdeploy.codebase.mmdet.models.utils.gaussian_target import \
        get_topk_from_heatmap__default
    scores = torch.rand(1, 2, 4, 4)

    gts = get_topk_from_heatmap(scores, k=20)
    outs = get_topk_from_heatmap__default(scores, k=20)

    for gt, out in zip(gts, outs):
        torch.testing.assert_allclose(gt, out)

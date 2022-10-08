# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import (WrapFunction, check_backend,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMDET)
except ImportError:
    pytest.skip(f'{Codebase.MMDET} is not installed.', allow_module_level=True)


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_distance2bbox(backend_type: Backend):
    check_backend(backend_type)
    deploy_cfg = Config(
        dict(
            onnx_config=dict(output_names=None, input_shape=None),
            backend_config=dict(type=backend_type.value, model_inputs=None),
            codebase_config=dict(type='mmdet', task='ObjectDetection')))

    # wrap function to enable rewrite
    def distance2bbox(*args, **kwargs):
        import mmdet.structures.bbox.transforms
        return mmdet.structures.bbox.transforms.distance2bbox(*args, **kwargs)

    points = torch.rand(3, 2)
    distance = torch.rand(3, 4)
    original_outputs = distance2bbox(points, distance)

    # wrap function to nn.Module, enable torch.onnx.export
    wrapped_func = WrapFunction(distance2bbox)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_func,
        model_inputs={
            'points': points,
            'distance': distance
        },
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        model_output = original_outputs.squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        assert rewrite_outputs is not None

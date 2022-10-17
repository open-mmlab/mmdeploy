# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict

import mmcv
import onnx
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.core import RewriterContext
from mmdeploy.utils import Backend, Codebase, get_onnx_config

try:
    import_codebase(Codebase.MMEDIT)
except ImportError:
    pytest.skip(
        f'{Codebase.MMEDIT} is not installed.', allow_module_level=True)

img = torch.rand(1, 3, 4, 4)
model_file = tempfile.NamedTemporaryFile(suffix='.onnx').name

deploy_cfg = mmcv.Config(
    dict(
        codebase_config=dict(
            type='mmedit',
            task='SuperResolution',
        ),
        backend_config=dict(
            type='tensorrt',
            common_config=dict(fp16_mode=False, max_workspace_size=1 << 10),
            model_inputs=[
                dict(
                    input_shapes=dict(
                        input=dict(
                            min_shape=[1, 3, 4, 4],
                            opt_shape=[1, 3, 4, 4],
                            max_shape=[1, 3, 4, 4])))
            ]),
        ir_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            save_file=model_file,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))


def test_srcnn():
    from mmedit.models.backbones.sr_backbones import SRCNN
    pytorch_model = SRCNN()
    model_inputs = {'x': img}

    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    onnx_cfg = get_onnx_config(deploy_cfg)
    input_names = [k for k, v in model_inputs.items() if k != 'ctx']

    dynamic_axes = onnx_cfg.get('dynamic_axes', None)

    if dynamic_axes is not None and not isinstance(dynamic_axes, Dict):
        dynamic_axes = zip(input_names, dynamic_axes)

    with RewriterContext(
            cfg=deploy_cfg, backend=Backend.TENSORRT.value), torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            tuple([v for k, v in model_inputs.items()]),
            onnx_file_path,
            export_params=True,
            input_names=input_names,
            output_names=None,
            opset_version=11,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=False)

    # The result should be different due to the rewrite.
    # So we only check if the file exists
    assert osp.exists(onnx_file_path)

    model = onnx.load(onnx_file_path)
    assert model is not None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        assert False

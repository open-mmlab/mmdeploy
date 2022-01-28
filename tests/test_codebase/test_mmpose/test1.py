# Copyright (c) OpenMMLab. All rights reserved.
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np

from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config

import_codebase(Codebase.MMPOSE)

model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmpose', task='PoseDetection'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            save_file='end2end.onnx',
            input_names=['input'],
            output_names=['output'],
            input_shape=None)))

onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
img_shape = (192, 256)
heatmap_shape = (48, 64)
# mmpose.apis.inference.LoadImage uses opencv, needs float32 in
# cv2.cvtColor.
img = np.random.rand(*img_shape, 3).astype(np.float32)
num_output_channels = model_cfg['data_cfg']['num_output_channels']


def test_create_input():
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2

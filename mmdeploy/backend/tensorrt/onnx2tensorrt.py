# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Union

import mmcv
import onnx

from mmdeploy.utils import (get_calib_filename, get_common_config,
                            get_model_inputs, load_config, parse_device_id)
from mmdeploy.utils.config_utils import get_ir_config
from .utils import from_onnx, get_trt_log_level


def onnx2tensorrt(work_dir: str,
                  save_file: str,
                  model_id: int,
                  deploy_cfg: Union[str, mmcv.Config],
                  onnx_model: Union[str, onnx.ModelProto],
                  device: str = 'cuda:0',
                  partition_type: str = 'end2end',
                  **kwargs):
    """Convert ONNX to TensorRT.

    Examples:
        >>> from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt
        >>> work_dir = 'work_dir'
        >>> save_file = 'end2end.engine'
        >>> model_id = 0
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_tensorrt_dynamic-320x320-1344x1344.py')
        >>> onnx_model = 'work_dir/end2end.onnx'
        >>> onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg,
                onnx_model, 'cuda:0')

    Args:
        work_dir (str): A working directory.
        save_file (str): The base name of the file to save TensorRT engine.
            E.g. `end2end.engine`.
        model_id (int): Index of input model.
        deploy_cfg (str | mmcv.Config): Deployment config.
        onnx_model (str | onnx.ModelProto): input onnx model.
        device (str): A string specifying cuda device, defaults to 'cuda:0'.
        partition_type (str): Specifying partition type of a model, defaults to
            'end2end'.
    """

    # load deploy_cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]

    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    common_params = get_common_config(deploy_cfg)
    model_params = get_model_inputs(deploy_cfg)[model_id]

    final_params = common_params
    final_params.update(model_params)

    int8_param = final_params.get('int8_param', dict())
    calib_file = get_calib_filename(deploy_cfg)
    if calib_file is not None:
        int8_param['calib_file'] = osp.join(work_dir, calib_file)
        int8_param['model_type'] = partition_type

    ir_config = get_ir_config(deploy_cfg)
    input_names = ir_config.get('input_names', [])
    input_shapes = final_params['input_shapes']

    if not isinstance(input_shapes, Dict):
        input_shapes = dict(zip(input_names, input_shapes))

    assert device.startswith('cuda'), 'TensorRT requires cuda device,' \
        f'but given: {device}'

    device_id = parse_device_id(device)
    assert save_file.endswith(
        '.engine'
    ), 'Expect save file ends with `.engine`.' f' but get {save_file}'
    save_path = osp.join(work_dir, save_file)
    from_onnx(
        onnx_model,
        osp.splitext(save_path)[0],
        input_shapes=input_shapes,
        log_level=get_trt_log_level(),
        fp16_mode=final_params.get('fp16_mode', False),
        int8_mode=final_params.get('int8_mode', False),
        int8_param=int8_param,
        max_workspace_size=final_params.get('max_workspace_size', 0),
        device_id=device_id)

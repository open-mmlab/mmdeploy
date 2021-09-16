import os.path as osp
from typing import Union

import mmcv
import onnx
import tensorrt as trt

from mmdeploy.utils import (get_calib_filename, get_common_config,
                            get_model_inputs, load_config)
from .tensorrt_utils import create_trt_engine, save_trt_engine


def parse_device_id(device: str):
    device_id = 0
    if len(device) >= 6:
        device_id = int(device[5:])
    return device_id


def onnx2tensorrt(work_dir: str,
                  save_file: str,
                  model_id: int,
                  deploy_cfg: Union[str, mmcv.Config],
                  onnx_model: Union[str, onnx.ModelProto],
                  device: str = 'cuda:0',
                  partition_type: str = 'end2end',
                  **kwargs):

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

    assert device.startswith('cuda')
    device_id = parse_device_id(device)
    engine = create_trt_engine(
        onnx_model,
        opt_shape_dict=final_params['input_shapes'],
        log_level=final_params.get('log_level', trt.Logger.WARNING),
        fp16_mode=final_params.get('fp16_mode', False),
        int8_mode=final_params.get('int8_mode', False),
        int8_param=int8_param,
        max_workspace_size=final_params.get('max_workspace_size', 0),
        device_id=device_id)

    save_trt_engine(engine, osp.join(work_dir, save_file))

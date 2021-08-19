import os.path as osp
from typing import Optional, Union

import mmcv
import onnx
import tensorrt as trt
import torch.multiprocessing as mp

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
                  split_type: str = 'end2end',
                  ret_value: Optional[mp.Value] = None,
                  **kwargs):
    ret_value.value = -1

    # load deploy_cfg if necessary
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    elif not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    assert 'tensorrt_params' in deploy_cfg

    tensorrt_params = deploy_cfg['tensorrt_params']
    shared_params = tensorrt_params.get('shared_params', dict())
    model_params = tensorrt_params['model_params'][model_id]

    final_params = shared_params
    final_params.update(model_params)

    int8_param = final_params.get('int8_param', dict())
    if deploy_cfg.get('create_calib', False):
        calib_params = deploy_cfg.get('calib_params', dict())
        calib_file = calib_params.get('calib_file', 'calib_file.h5')
        int8_param['calib_file'] = osp.join(work_dir, calib_file)
        int8_param['model_type'] = split_type

    assert device.startswith('cuda')
    device_id = parse_device_id(device)
    engine = create_trt_engine(
        onnx_model,
        opt_shape_dict=final_params['opt_shape_dict'],
        log_level=final_params.get('log_level', trt.Logger.WARNING),
        fp16_mode=final_params.get('fp16_mode', False),
        int8_mode=final_params.get('int8_mode', False),
        int8_param=int8_param,
        max_workspace_size=final_params.get('max_workspace_size', 0),
        device_id=device_id)

    save_trt_engine(engine, osp.join(work_dir, save_file))

    ret_value.value = 0

import logging
import os.path as osp
from typing import Optional, Union

import mmcv
import onnx
import tensorrt as trt
import torch.multiprocessing as mp

from .tensorrt_utils import create_trt_engine, save_trt_engine


def get_trt_loglevel():
    logger = logging.getLogger()
    level = logger.level

    if level == logging.INFO:
        return trt.Logger.INFO
    elif level == logging.ERROR or level == logging.CRITICAL:
        return trt.Logger.ERROR
    elif level == logging.WARNING:
        return trt.Logger.WARNING
    else:
        print('for logging level: {}, use trt.Logger.INFO'.format(level))
        return trt.Logger.INFO


def onnx2tensorrt(work_dir: str,
                  save_file: str,
                  model_id: int,
                  deploy_cfg: Union[str, mmcv.Config],
                  onnx_model: Union[str, onnx.ModelProto],
                  device: str = 'cuda:0',
                  ret_value: Optional[mp.Value] = None):
    ret_value.value = -1

    # load deploy_cfg if necessary
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    elif not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    assert 'tensorrt_param' in deploy_cfg

    tensorrt_param = deploy_cfg['tensorrt_param']
    shared_param = tensorrt_param.get('shared_param', dict())
    model_param = tensorrt_param['model_params'][model_id]

    final_param = shared_param
    final_param.update(model_param)

    assert device.startswith('cuda')
    device_id = 0
    if len(device) >= 6:
        device_id = int(device[5:])
    engine = create_trt_engine(
        onnx_model,
        opt_shape_dict=final_param['opt_shape_dict'],
        log_level=final_param.get('log_level', get_trt_loglevel()),
        fp16_mode=final_param.get('fp16_mode', False),
        max_workspace_size=final_param.get('max_workspace_size', 0),
        device_id=device_id)

    save_trt_engine(engine, osp.join(work_dir, save_file))

    ret_value.value = 0

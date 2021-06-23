import os.path as osp
from typing import Optional, Union

import tensorrt as trt

import mmcv
import onnx
import torch.multiprocessing as mp

from .tensorrt_utils import onnx2trt, save_trt_engine


def onnx2tensorrt(work_dir: str,
                  save_file: str,
                  deploy_cfg: Union[str, mmcv.Config],
                  onnx_model: Union[str, onnx.ModelProto],
                  device: str = 'cuda:0',
                  ret_value: Optional[mp.Value] = None):
    ret_value.value = -1
    save_file = 'onnx2tensorrt.engine'

    # load deploy_cfg if needed
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    elif not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    assert 'tensorrt_param' in deploy_cfg

    tensorrt_param = deploy_cfg['tensorrt_param']

    assert device.startswith('cuda')
    device_id = 0
    if len(device) >= 6:
        device_id = int(device[5:])
    engine = onnx2trt(
        onnx_model,
        opt_shape_dict=tensorrt_param['opt_shape_dict'],
        log_level=tensorrt_param.get('log_level', trt.Logger.WARNING),
        fp16_mode=tensorrt_param.get('fp16_mode', False),
        max_workspace_size=tensorrt_param.get('max_workspace_size', 0),
        device_id=device_id)

    save_trt_engine(engine, osp.join(work_dir, save_file))

    ret_value.value = 0

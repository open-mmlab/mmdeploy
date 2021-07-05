import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch
import torch.multiprocessing as mp

from mmdeploy.utils import (RewriterContext, patch_model,
                            register_extra_symbolics)
from .utils import create_input, init_model


def torch2onnx_impl(model: torch.nn.Module, input: torch.Tensor,
                    deploy_cfg: Union[str, mmcv.Config], output_file: str):
    # load deploy_cfg if needed
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    pytorch2onnx_cfg = deploy_cfg['pytorch2onnx']
    backend = deploy_cfg['backend']
    opset_version = pytorch2onnx_cfg.get('opset_version', 11)

    # load registed symbolic
    register_extra_symbolics(deploy_cfg, backend=backend, opset=opset_version)

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with RewriterContext(cfg=deploy_cfg, backend=backend):
        torch.onnx.export(
            patched_model,
            input,
            output_file,
            export_params=pytorch2onnx_cfg['export_params'],
            input_names=pytorch2onnx_cfg['input_names'],
            output_names=pytorch2onnx_cfg['output_names'],
            opset_version=opset_version,
            dynamic_axes=pytorch2onnx_cfg.get('dynamic_axes', None),
            keep_initializers_as_inputs=pytorch2onnx_cfg[
                'keep_initializers_as_inputs'])


def torch2onnx(img: Any,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               ret_value: Optional[mp.Value] = None):

    if ret_value is not None:
        ret_value.value = -1

    # load deploy_cfg if needed
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')
    # load model_cfg if needed
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    if not isinstance(model_cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    output_file = osp.join(work_dir, save_file)

    codebase = deploy_cfg['codebase']

    torch_model = init_model(codebase, model_cfg, model_checkpoint, device)
    data, model_inputs = create_input(codebase, model_cfg, img, device)
    if not isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs[0]

    torch2onnx_impl(
        torch_model,
        model_inputs,
        deploy_cfg=deploy_cfg,
        output_file=output_file)

    if ret_value is not None:
        ret_value.value = 0

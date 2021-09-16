import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch

from mmdeploy.core import (RewriterContext, patch_model,
                           register_extra_symbolics)
from mmdeploy.utils import (get_backend, get_codebase, get_input_shape,
                            get_onnx_config, get_task_type, load_config)
from .utils import create_input, init_pytorch_model


def torch2onnx_impl(model: torch.nn.Module, input: torch.Tensor,
                    deploy_cfg: Union[str, mmcv.Config], output_file: str):
    # load deploy_cfg if needed
    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    if not isinstance(deploy_cfg, mmcv.Config):
        raise TypeError('deploy_cfg must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    pytorch2onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg).value
    opset_version = pytorch2onnx_cfg.get('opset_version', 11)

    # load registed symbolic
    register_extra_symbolics(deploy_cfg, backend=backend, opset=opset_version)

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with RewriterContext(cfg=deploy_cfg, backend=backend), torch.no_grad():
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
               device: str = 'cuda:0'):

    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    output_file = osp.join(work_dir, save_file)

    codebase = get_codebase(deploy_cfg)
    task = get_task_type(deploy_cfg)
    input_shape = get_input_shape(deploy_cfg)

    torch_model = init_pytorch_model(codebase, model_cfg, model_checkpoint,
                                     device)
    data, model_inputs = create_input(codebase, task, model_cfg, img,
                                      input_shape, device)
    if not isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs[0]

    torch2onnx_impl(
        torch_model,
        model_inputs,
        deploy_cfg=deploy_cfg,
        output_file=output_file)

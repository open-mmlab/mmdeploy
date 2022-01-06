# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import (get_backend, get_dynamic_axes, get_input_shape,
                            get_onnx_config, load_config)


def torch2onnx_impl(model: torch.nn.Module, input: torch.Tensor,
                    deploy_cfg: Union[str, mmcv.Config], output_file: str):
    """Converting torch model to ONNX.

    Args:
        model (torch.nn.Module): Input pytorch model.
        input (torch.Tensor): Input tensor used to convert model.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        output_file (str): Output file to save ONNX model.
    """
    # load deploy_cfg if needed
    deploy_cfg = load_config(deploy_cfg)[0]

    onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg).value
    opset_version = onnx_cfg.get('opset_version', 11)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with RewriterContext(
            cfg=deploy_cfg, backend=backend,
            opset=opset_version), torch.no_grad():
        torch.onnx.export(
            patched_model,
            input,
            output_file,
            export_params=onnx_cfg['export_params'],
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=onnx_cfg[
                'keep_initializers_as_inputs'],
            strip_doc_string=onnx_cfg.get('strip_doc_string', True))


def torch2onnx(img: Any,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0'):
    """Convert PyTorch model to ONNX model.

    Args:
        img (str | np.ndarray | torch.Tensor): Input image used to assist
            converting model.
        work_dir (str): A working directory to save files.
        save_file (str): Filename to save onnx model.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        device (str): A string specifying device type, defaults to 'cuda:0'.
    """
    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    output_file = osp.join(work_dir, save_file)

    input_shape = get_input_shape(deploy_cfg)

    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = task_processor.init_pytorch_model(model_checkpoint)
    data, model_inputs = task_processor.create_input(img, input_shape)
    if not isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs[0]

    torch2onnx_impl(
        torch_model,
        model_inputs,
        deploy_cfg=deploy_cfg,
        output_file=output_file)

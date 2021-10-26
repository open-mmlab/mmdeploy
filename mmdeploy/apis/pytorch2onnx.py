import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import (get_backend, get_codebase, get_input_shape,
                            get_onnx_config, get_task_type, load_config)
from .utils import create_input, init_pytorch_model


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

    pytorch2onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg).value
    opset_version = pytorch2onnx_cfg.get('opset_version', 11)

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with RewriterContext(
            cfg=deploy_cfg, backend=backend,
            opset=opset_version), torch.no_grad():
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
                'keep_initializers_as_inputs'],
            strip_doc_string=pytorch2onnx_cfg.get('strip_doc_string', True))


def torch2onnx(img: Any,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0'):
    """Convert PyToch model to ONNX model.

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

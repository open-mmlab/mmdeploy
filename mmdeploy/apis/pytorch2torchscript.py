import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import get_backend, get_input_shape, load_config


def torch2torchscript_impl(model: torch.nn.Module, input: torch.Tensor,
                           deploy_cfg: Union[str,
                                             mmcv.Config], output_file: str):
    """Converting torch model to torchscript.

    Args:
        model (torch.nn.Module): Input pytorch model.
        input (torch.Tensor): Input tensor used to convert model.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        output_file (str): Output file to save torchscript model.
    """
    deploy_cfg = load_config(deploy_cfg)[0]

    # ir_cfg = get_ir_config(deploy_cfg)
    backend = get_backend(deploy_cfg).value

    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend)

    with RewriterContext(
            cfg=deploy_cfg,
            backend=backend), torch.no_grad(), torch.jit.optimized_execution(
                True):
        ts_model = torch.jit.trace(patched_model, input)

    # TODO: custom optimize

    torch.jit.save(ts_model, output_file)


def torch2torchscript(img: Any,
                      work_dir: str,
                      save_file: str,
                      deploy_cfg: Union[str, mmcv.Config],
                      model_cfg: Union[str, mmcv.Config],
                      model_checkpoint: Optional[str] = None,
                      device: str = 'cuda:0'):
    """Convert PyTorch model to torchscript model.

    Args:
        img (str | np.ndarray | torch.Tensor): Input image used to assist
            converting model.
        work_dir (str): A working directory to save files.
        save_file (str): Filename to save torchscript model.
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
    _, model_inputs = task_processor.create_input(img, input_shape)
    if not isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs[0]

    torch2torchscript_impl(
        torch_model,
        model_inputs,
        deploy_cfg=deploy_cfg,
        output_file=output_file)

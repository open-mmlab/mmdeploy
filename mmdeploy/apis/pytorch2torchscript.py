# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Union

import mmcv

from mmdeploy.apis.core.pipeline_manager import PIPELINE_MANAGER, no_mp


@PIPELINE_MANAGER.register_pipeline()
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
    import torch

    from mmdeploy.utils import get_backend, get_input_shape, load_config
    from .torch_jit import trace

    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    input_shape = get_input_shape(deploy_cfg)

    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = task_processor.init_pytorch_model(model_checkpoint)
    _, model_inputs = task_processor.create_input(img, input_shape)
    if not isinstance(model_inputs, torch.Tensor):
        model_inputs = model_inputs[0]

    context_info = dict(deploy_cfg=deploy_cfg)
    backend = get_backend(deploy_cfg).value
    output_prefix = osp.join(work_dir, osp.splitext(save_file)[0])

    with no_mp():
        trace(
            torch_model,
            model_inputs,
            output_path_prefix=output_prefix,
            backend=backend,
            context_info=context_info,
            check_trace=False)

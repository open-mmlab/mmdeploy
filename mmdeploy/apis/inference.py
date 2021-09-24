from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmdeploy.utils import (Backend, get_backend, get_codebase,
                            get_input_shape, get_task_type, load_config)
from .utils import (create_input, init_backend_model, init_pytorch_model,
                    run_inference, visualize)


def inference_model(model_cfg: Union[str, mmcv.Config],
                    deploy_cfg: Union[str, mmcv.Config],
                    model: Union[str, Sequence[str], torch.nn.Module],
                    img: Union[str, np.ndarray],
                    device: str,
                    backend: Optional[Backend] = None,
                    output_file: Optional[str] = None,
                    show_result: bool = False):
    """Run inference with PyTorch or backend model and show results.

    Args:
        model_cfg (str | mmcv.Config): Model config file or Config object.
        deploy_cfg (str | mmcv.Config): Deployment config file or Config
            object.
        model (str | list[str], torch.nn.Module): Input model or file(s).
        img (str | np.ndarray): Input image file or numpy array for inference.
        device (str): A string specifying device type.
        backend (Backend): Specifying backend type, defaults to `None`.
        output_file (str): Output file to save visualized image, defaults to
            `None`. Only valid if `show_result` is set to `False`.
        show_result (bool): Whether to show plotted image in windows, defaults
            to `False`.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    codebase = get_codebase(deploy_cfg)
    task = get_task_type(deploy_cfg)
    input_shape = get_input_shape(deploy_cfg)
    if backend is None:
        backend = get_backend(deploy_cfg)

    if isinstance(model, str):
        model = [model]

    if isinstance(model, (list, tuple)):
        assert len(model) > 0, 'Model should have at least one element.'
        assert all([isinstance(m, str) for m in model]), 'All elements in the \
            list should be str'

        if backend == Backend.PYTORCH:
            model = init_pytorch_model(codebase, model_cfg, model[0], device)
        else:
            device_id = -1 if device == 'cpu' else 0
            model = init_backend_model(
                model,
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg,
                device_id=device_id)

    model_inputs, _ = create_input(codebase, task, model_cfg, img, input_shape,
                                   device)

    with torch.no_grad():
        result = run_inference(codebase, model_inputs, model)

    visualize(
        codebase,
        img,
        result=result,
        model=model,
        output_file=output_file,
        backend=backend,
        show_result=show_result)

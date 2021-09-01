from typing import Optional

import torch

from mmdeploy.utils import Backend, get_backend, get_codebase, load_config
from .utils import (create_input, init_backend_model, init_pytorch_model,
                    run_inference, visualize)


def inference_model(model_cfg,
                    deploy_cfg,
                    model,
                    img,
                    device: str,
                    backend: Optional[Backend] = None,
                    output_file: Optional[str] = None,
                    show_result=False):

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    codebase = get_codebase(deploy_cfg)
    if backend is None:
        backend = get_backend(deploy_cfg)

    if isinstance(model, str):
        model = [model]
    if isinstance(model, (list, tuple)):
        if backend == Backend.PYTORCH:
            model = init_pytorch_model(codebase, model_cfg, model[0], device)
        else:
            device_id = -1 if device == 'cpu' else 0
            model = init_backend_model(
                model,
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg,
                device_id=device_id)

    model_inputs, _ = create_input(codebase, model_cfg, img, device)

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

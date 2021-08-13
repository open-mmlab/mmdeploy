from typing import Optional

import torch.multiprocessing as mp

from .utils import (assert_cfg_valid, check_model_outputs, create_input,
                    init_backend_model, init_model)


def inference_model(model_cfg,
                    deploy_cfg,
                    model,
                    img,
                    device: str,
                    backend: Optional[str] = None,
                    output_file: Optional[str] = None,
                    show_result=False,
                    ret_value: Optional[mp.Value] = None):

    if ret_value is not None:
        ret_value.value = -1

    deploy_cfg, model_cfg = assert_cfg_valid(deploy_cfg, model_cfg)

    codebase = deploy_cfg['codebase']
    if backend is None:
        backend = deploy_cfg['backend']

    if isinstance(model, str):
        model = [model]
    if isinstance(model, (list, tuple)):
        if backend == 'pytorch':
            model = init_model(codebase, model_cfg, model[0], device)
        else:
            device_id = -1 if device == 'cpu' else 0
            model = init_backend_model(
                model,
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg,
                device_id=device_id)

    model_inputs, _ = create_input(codebase, model_cfg, img, device)

    check_model_outputs(
        codebase,
        img,
        model_inputs=model_inputs,
        model=model,
        output_file=output_file,
        backend=backend,
        show_result=show_result)

    if ret_value is not None:
        ret_value.value = 0

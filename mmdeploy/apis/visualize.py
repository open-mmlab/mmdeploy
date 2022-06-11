# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmdeploy.codebase import BaseTask
from mmdeploy.utils import Backend, get_backend, get_input_shape, load_config


def visualize_model(model_cfg: Union[str, mmcv.Config],
                    deploy_cfg: Union[str, mmcv.Config],
                    model: Union[str, Sequence[str], BaseTask],
                    img: Union[str, np.ndarray],
                    device: str,
                    backend: Optional[Backend] = None,
                    output_file: Optional[str] = None,
                    show_result: bool = False):
    """Run inference with PyTorch or backend model and show results.

    Examples:
        >>> from mmdeploy.apis import visualize_model
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> model = 'work_dir/fcos.onnx'
        >>> img = 'demo.jpg'
        >>> device = 'cpu'
        >>> visualize_model(model_cfg, deploy_cfg, model, \
            img, device, show_result=True)

    Args:
        model_cfg (str | mmcv.Config): Model config file or Config object.
        deploy_cfg (str | mmcv.Config): Deployment config file or Config
            object.
        model (str | list[str], BaseSubtask): Input model or file(s).
        img (str | np.ndarray): Input image file or numpy array for inference.
        device (str): A string specifying device type.
        backend (Backend): Specifying backend type, defaults to `None`.
        output_file (str): Output file to save visualized image, defaults to
            `None`. Only valid if `show_result` is set to `False`.
        show_result (bool): Whether to show plotted image in windows, defaults
            to `False`.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    from mmdeploy.apis.utils import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

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
            model = task_processor.init_pytorch_model(model[0])
        else:
            model = task_processor.init_backend_model(model)

    model_inputs, _ = task_processor.create_input(img, input_shape)
    with torch.no_grad():
        result = task_processor.run_inference(model, model_inputs)[0]

    task_processor.visualize(
        image=img,
        model=model,
        result=result,
        output_file=output_file,
        window_name=backend.value,
        show_result=show_result)

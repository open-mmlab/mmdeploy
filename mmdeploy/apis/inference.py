# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence, Union

import mmcv
import numpy as np


def inference_model(model_cfg: Union[str, mmcv.Config],
                    deploy_cfg: Union[str, mmcv.Config],
                    backend_files: Sequence[str], img: Union[str, np.ndarray],
                    device: str) -> Any:
    """Run inference with PyTorch or backend model and show results.

    Examples:
        >>> from mmdeploy.apis import inference_model
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> backend_files = ['work_dir/fcos.onnx']
        >>> img = 'demo.jpg'
        >>> device = 'cpu'
        >>> model_output = inference_model(model_cfg, deploy_cfg,
                            backend_files, img, device)

    Args:
        model_cfg (str | mmcv.Config): Model config file or Config object.
        deploy_cfg (str | mmcv.Config): Deployment config file or Config
            object.
        backend_files (Sequence[str]): Input backend model file(s).
        img (str | np.ndarray): Input image file or numpy array for inference.
        device (str): A string specifying device type.

    Returns:
        Any: The inference results
    """
    import torch

    from mmdeploy.utils import get_input_shape, load_config

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    from mmdeploy.apis.utils import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    model = task_processor.init_backend_model(backend_files)

    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(img, input_shape)

    with torch.no_grad():
        result = task_processor.run_inference(model, model_inputs)

    return result

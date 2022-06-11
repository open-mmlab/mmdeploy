# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Union

import mmcv
import torch

from mmdeploy.apis.core.pipeline_manager import no_mp
from mmdeploy.utils import (get_backend, get_dynamic_axes, get_input_shape,
                            get_onnx_config, load_config)
from .core import PIPELINE_MANAGER
from .onnx import export


@PIPELINE_MANAGER.register_pipeline()
def torch2onnx(img: Any,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmcv.Config],
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0'):
    """Convert PyTorch model to ONNX model.

    Examples:
        >>> from mmdeploy.apis import torch2onnx
        >>> img = 'demo.jpg'
        >>> work_dir = 'work_dir'
        >>> save_file = 'fcos.onnx'
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> model_checkpoint = ('checkpoints/'
                                'fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth')
        >>> device = 'cpu'
        >>> torch2onnx(img, work_dir, save_file, deploy_cfg, \
            model_cfg, model_checkpoint, device)

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

    input_shape = get_input_shape(deploy_cfg)

    # create model an inputs
    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = task_processor.init_pytorch_model(model_checkpoint)
    data, model_inputs = task_processor.create_input(img, input_shape)
    if not isinstance(model_inputs, torch.Tensor) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]

    # export to onnx
    context_info = dict()
    context_info['deploy_cfg'] = deploy_cfg
    output_prefix = osp.join(work_dir,
                             osp.splitext(osp.basename(save_file))[0])
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get('opset_version', 11)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
        'verbose', False)
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs',
                                               True)
    with no_mp():
        export(
            torch_model,
            model_inputs,
            output_path_prefix=output_prefix,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs)

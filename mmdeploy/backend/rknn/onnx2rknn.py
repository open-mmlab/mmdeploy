# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Union, Optional

import mmcv
from rknn.api import RKNN

from mmdeploy.utils import (load_config, get_common_config, get_root_logger,
                            get_onnx_config, get_quantization_config,
                            get_model_inputs)
from mmdeploy.utils.config_utils import get_backend_config


def onnx2rknn(onnx_model: str,
              output_path: str,
              deploy_cfg: Union[str, mmcv.Config],
              dataset_file: Optional[str] = None,
              **kwargs):
    """Convert ONNX to RKNN.

    RKNN-Toolkit2 is a software development kit for users to perform model
    conversion, inference and performance evaluation on PC and Rockchip
    NPU platforms.

    Args:
        onnx_model (str): Input onnx model.
        output_path (str): File path to save RKNN model.
        device (str): A string specifying device, defaults to 'cuda:0'.
        input_shapes (Sequence[Sequence[int]] | None): Shapes for PPLNN
            optimization, default to None.

    Examples:
        >>> from mmdeploy.apis.rknn import from_onnx
        >>>
        >>> from_onnx(onnx_model = 'example.onnx',
                      output_file_prefix = 'example')
    """
    logger = get_root_logger()
    # load deploy_cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]

    common_params = get_common_config(deploy_cfg)
    # common_params.update(dict(mean_values=[0, 0, 0], std_values=[1, 1, 1]))
    onnx_params = get_onnx_config(deploy_cfg)
    quantization_cfg = get_quantization_config(deploy_cfg)

    input_names = onnx_params.get('input_names', None)
    output_names = onnx_params.get('output_names', None)
    input_size_list = get_backend_config(deploy_cfg).get(
        'input_size_list', None)

    rknn = RKNN(verbose=True)
    rknn.config(**common_params)
    ret = rknn.load_onnx(
        model=onnx_model,
        inputs=input_names,
        input_size_list=input_size_list,
        outputs=['pred_maps.0', 'pred_maps.1', 'pred_maps.2'])
    if ret != 0:
        logger.error('Load model failed!')
        exit(ret)

    dataset_cfg = quantization_cfg.get('dataset', None)
    do_quantization = quantization_cfg.get('do_quantization', False)
    if dataset_cfg is None and dataset_file is None:
        do_quantization = False
        logger.warning('no dataset passed in, quantization is skipped')
    if dataset_file is None:
        dataset_file = dataset_cfg
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_file)
    if ret != 0:
        logger.error('Build model failed!')
        exit(ret)

    ret = rknn.export_rknn(output_path)
    if ret != 0:
        logger.error('Export rknn model failed!')
        exit(ret)

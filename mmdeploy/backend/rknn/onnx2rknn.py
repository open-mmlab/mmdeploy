# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
from rknn.api import RKNN

from mmdeploy.utils import (get_backend_config, get_common_config,
                            get_normalization, get_onnx_config,
                            get_partition_config, get_quantization_config,
                            get_rknn_quantization, get_root_logger,
                            load_config)


def onnx2rknn(onnx_model: str,
              output_path: str,
              deploy_cfg: Union[str, mmcv.Config],
              model_cfg: Optional[Union[str, mmcv.Config]] = None,
              dataset_file: Optional[str] = None,
              **kwargs):
    """Convert ONNX to RKNN.

    RKNN-Toolkit2 is a software development kit for users to perform model
    conversion, inference and performance evaluation on PC and Rockchip
    NPU platforms.

    Args:
        onnx_model (str): Input onnx model.
        output_path (str): File path to save RKNN model.
        deploy_cfg (str | mmcv.Config): The path or content of config.
        dataset_file (str | None): The dataset file for quatization. Default to
            None.
    """
    logger = get_root_logger()
    # load deploy_cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]

    common_params = get_common_config(deploy_cfg)
    onnx_params = get_onnx_config(deploy_cfg)
    quantization_cfg = get_quantization_config(deploy_cfg)

    input_names = onnx_params.get('input_names', None)
    output_names = onnx_params.get('output_names', None)
    input_size_list = get_backend_config(deploy_cfg).get(
        'input_size_list', None)
    # update norm value
    if get_rknn_quantization(deploy_cfg) is True and model_cfg is not None:
        transform = get_normalization(model_cfg)
        common_params.update(
            dict(
                mean_values=[transform['mean']],
                std_values=[transform['std']]))

    # update output_names for partition models
    if get_partition_config(deploy_cfg) is not None:
        import onnx
        _onnx_model = onnx.load(onnx_model)
        output_names = [node.name for node in _onnx_model.graph.output]

    rknn = RKNN(verbose=True)
    rknn.config(**common_params)
    ret = rknn.load_onnx(
        model=onnx_model,
        inputs=input_names,
        input_size_list=input_size_list,
        outputs=output_names)
    if ret != 0:
        logger.error('Load model failed!')
        exit(ret)

    dataset_cfg = quantization_cfg.get('dataset', None)
    do_quantization = quantization_cfg.get('do_quantization', False)
    if dataset_cfg is None and dataset_file is None:
        do_quantization = False
        logger.warning('no dataset passed in, quantization is skipped')
    if dataset_cfg is not None:
        dataset_file = dataset_cfg
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_file)
    if ret != 0:
        logger.error('Build model failed!')
        exit(ret)

    ret = rknn.export_rknn(output_path)
    if ret != 0:
        logger.error('Export rknn model failed!')
        exit(ret)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.ir.onnx import ONNXManager
from mmdeploy.utils import Backend


@PIPELINE_MANAGER.register_pipeline()
def export(model: torch.nn.Module,
           args: Union[torch.Tensor, Tuple, Dict],
           output_path_prefix: str,
           backend: Union[Backend, str] = 'default',
           input_metas: Optional[Dict] = None,
           context_info: Dict = dict(),
           input_names: Optional[Sequence[str]] = None,
           output_names: Optional[Sequence[str]] = None,
           opset_version: int = 11,
           dynamic_axes: Optional[Dict] = None,
           verbose: bool = False,
           keep_initializers_as_inputs: Optional[bool] = None,
           optimize: bool = False):
    """Export a PyTorch model into ONNX format. This is a wrap of
    `torch.onnx.export` with some enhancement.

    Examples:
        >>> from mmdeploy.apis.onnx import export
        >>>
        >>> model = create_model()
        >>> args = get_input_tensor()
        >>>
        >>> export(
        >>>     model,
        >>>     args,
        >>>     'place/to/save/model',
        >>>     backend='tensorrt',
        >>>     input_names=['input'],
        >>>     output_names=['output'],
        >>>     dynamic_axes={'input': {
        >>>         0: 'batch',
        >>>         2: 'height',
        >>>         3: 'width'
        >>>     }})

    Args:
        model (torch.nn.Module): the model to be exported.
        args (torch.Tensor|Tuple|Dict): Dummy input of the model.
        output_path_prefix (str): The output file prefix. The model will
            be saved to `<output_path_prefix>.onnx`.
        backend (Backend|str): Which backend will the graph be used. Different
            backend would generate different graph.
        input_metas (Dict): The constant inputs of the model.
        context_info (Dict): The information that would be used in the context
            of exporting.
        input_names (Sequence[str]): The input names of the model.
        output_names (Sequence[str]): The output names of the model.
        opset_version (int): The version of ONNX opset version. 11 as default.
        dynamic_axes (Dict): The information used to determine which axes are
            dynamic.
        verbose (bool): Enable verbose model on `torch.onnx.export`.
        keep_initializers_as_inputs (bool): Whether we should add inputs for
            each initializer.
        optimize (bool): Perform optimize on model.
    """
    output_path = output_path_prefix + '.onnx'

    deploy_cfg = context_info.pop('deploy_cfg', dict())
    ONNXManager.export(
        model,
        args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        backend=backend,
        const_args=input_metas,
        rewrite_context=deploy_cfg,
        optimize=optimize)

# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Iterable, Optional, Union

import onnx

from .core import PIPELINE_MANAGER


@PIPELINE_MANAGER.register_pipeline()
def extract_model(model: Union[str, onnx.ModelProto],
                  start_marker: Union[str, Iterable[str]],
                  end_marker: Union[str, Iterable[str]],
                  start_name_map: Optional[Dict[str, str]] = None,
                  end_name_map: Optional[Dict[str, str]] = None,
                  dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                  save_file: Optional[str] = None) -> onnx.ModelProto:
    """Extract partition-model from an ONNX model.

    The partition-model is defined by the names of the input and output tensors
    exactly.

    Examples:
        >>> from mmdeploy.apis import extract_model
        >>> model = 'work_dir/fastrcnn.onnx'
        >>> start_marker = 'detector:input'
        >>> end_marker = ['extract_feat:output', 'multiclass_nms[0]:input']
        >>> dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'scores': {
                0: 'batch',
                1: 'num_boxes',
            },
            'boxes': {
                0: 'batch',
                1: 'num_boxes',
            }
        }
        >>> save_file = 'partition_model.onnx'
        >>> extract_model(model, start_marker, end_marker, \
                dynamic_axes=dynamic_axes, \
                save_file=save_file)

    Args:
        model (str | onnx.ModelProto): Input ONNX model to be extracted.
        start_marker (str | Sequence[str]): Start marker(s) to extract.
        end_marker (str | Sequence[str]): End marker(s) to extract.
        start_name_map (Dict[str, str]): A mapping of start names, defaults to
            `None`.
        end_name_map (Dict[str, str]): A mapping of end names, defaults to
            `None`.
        dynamic_axes (Dict[str, Dict[int, str]]): A dictionary to specify
            dynamic axes of input/output, defaults to `None`.
        save_file (str): A file to save the extracted model, defaults to
            `None`.

    Returns:
        onnx.ModelProto: The extracted model.
    """
    from .onnx import extract_partition

    return extract_partition(model, start_marker, end_marker, start_name_map,
                             end_name_map, dynamic_axes, save_file)

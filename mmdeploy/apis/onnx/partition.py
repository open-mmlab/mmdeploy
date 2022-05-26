# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterable, Optional, Union

import onnx
import onnx.helper
import onnx.utils

from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.core.optimizers import (attribute_to_dict, create_extractor,
                                      get_new_name, parse_extractor_io_string,
                                      remove_identity, rename_value)
from mmdeploy.utils import get_root_logger


@PIPELINE_MANAGER.register_pipeline()
def extract_partition(model: Union[str, onnx.ModelProto],
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
        >>> extract_partition(model, start_marker, end_marker, \
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
    if isinstance(model, str):
        model = onnx.load(model)

    num_value_info = len(model.graph.value_info)
    inputs = []
    outputs = []
    logger = get_root_logger()
    if not isinstance(start_marker, (list, tuple)):
        start_marker = [start_marker]
    for s in start_marker:
        start_name, func_id, start_type = parse_extractor_io_string(s)
        for node in model.graph.node:
            if node.op_type == 'Mark':
                attr = attribute_to_dict(node.attribute)
                if attr['func'] == start_name and attr[
                        'type'] == start_type and attr['func_id'] == func_id:
                    name = node.input[0]
                    if name not in inputs:
                        new_name = get_new_name(
                            attr, mark_name=s, name_map=start_name_map)
                        rename_value(model, name, new_name)
                        if not any([
                                v_info.name == new_name
                                for v_info in model.graph.value_info
                        ]):
                            new_val_info = onnx.helper.make_tensor_value_info(
                                new_name, attr['dtype'], attr['shape'])
                            model.graph.value_info.append(new_val_info)
                        inputs.append(new_name)

    logger.info(f'inputs: {", ".join(inputs)}')

    # collect outputs
    if not isinstance(end_marker, (list, tuple)):
        end_marker = [end_marker]
    for e in end_marker:
        end_name, func_id, end_type = parse_extractor_io_string(e)
        for node in model.graph.node:
            if node.op_type == 'Mark':
                attr = attribute_to_dict(node.attribute)
                if attr['func'] == end_name and attr[
                        'type'] == end_type and attr['func_id'] == func_id:
                    name = node.output[0]
                    if name not in outputs:
                        new_name = get_new_name(
                            attr, mark_name=e, name_map=end_name_map)
                        rename_value(model, name, new_name)
                        if not any([
                                v_info.name == new_name
                                for v_info in model.graph.value_info
                        ]):
                            new_val_info = onnx.helper.make_tensor_value_info(
                                new_name, attr['dtype'], attr['shape'])
                            model.graph.value_info.append(new_val_info)
                        outputs.append(new_name)

    logger.info(f'outputs: {", ".join(outputs)}')

    # replace Mark with Identity
    for node in model.graph.node:
        if node.op_type == 'Mark':
            del node.attribute[:]
            node.domain = ''
            node.op_type = 'Identity'

    extractor = create_extractor(model)
    extracted_model = extractor.extract_model(inputs, outputs)

    # remove all Identity, this may be done by onnx simplifier
    remove_identity(extracted_model)

    # collect all used inputs
    used = set()
    for node in extracted_model.graph.node:
        for input in node.input:
            used.add(input)

    for output in extracted_model.graph.output:
        used.add(output.name)

    # delete unused inputs
    success = True
    while success:
        success = False
        for i, input in enumerate(extracted_model.graph.input):
            if input.name not in used:
                del extracted_model.graph.input[i]
                success = True
                break

    # eliminate output without shape
    for xs in [extracted_model.graph.output]:
        for x in xs:
            if not x.type.tensor_type.shape.dim:
                logger.info(f'fixing output shape: {x.name}')
                x.CopyFrom(
                    onnx.helper.make_tensor_value_info(
                        x.name, x.type.tensor_type.elem_type, []))

    # eliminate 0-batch dimension, dirty workaround for two-stage detectors
    for input in extracted_model.graph.input:
        if input.name in inputs:
            if input.type.tensor_type.shape.dim[0].dim_value == 0:
                input.type.tensor_type.shape.dim[0].dim_value = 1

    # eliminate duplicated value_info for inputs
    success = True
    # num_value_info == 0 if dynamic shape
    if num_value_info == 0:
        while len(extracted_model.graph.value_info) > 0:
            extracted_model.graph.value_info.pop()
    while success:
        success = False
        for i, x in enumerate(extracted_model.graph.value_info):
            if x.name in inputs:
                del extracted_model.graph.value_info[i]
                success = True
                break

    # dynamic shape support
    if dynamic_axes is not None:
        for input_node in extracted_model.graph.input:
            if input_node.name in dynamic_axes:
                axes = dynamic_axes[input_node.name]
                for k, v in axes.items():
                    input_node.type.tensor_type.shape.dim[k].dim_value = 0
                    input_node.type.tensor_type.shape.dim[k].dim_param = v
        for output_node in extracted_model.graph.output:
            for idx, dim in enumerate(output_node.type.tensor_type.shape.dim):
                dim.dim_value = 0
                dim.dim_param = f'dim_{idx}'

    # save extract_model if save_file is given
    if save_file is not None:
        onnx.save(extracted_model, save_file)

    return extracted_model

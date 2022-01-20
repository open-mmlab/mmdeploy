# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Iterable, Optional

import onnx
from onnx.helper import get_attribute_value

from mmdeploy.utils import get_root_logger


def attribute_to_dict(attr: onnx.AttributeProto) -> Dict:
    """Convert onnx op attribute to dict.

    Args:
        attr (onnx.AttributeProto): Input onnx op attribute.

    Returns:
        dict: A dict contains info from op attribute.
    """
    ret = {}
    for a in attr:
        value = get_attribute_value(a)
        if isinstance(value, bytes):
            value = str(value, 'utf-8')
        ret[a.name] = value
    return ret


def remove_nodes(model: onnx.ModelProto,
                 predicate: Callable) -> onnx.ModelProto:
    """Remove nodes from ONNX model.

    Args:
        model (onnx.ModelProto): Input onnx model.
        predicate (Callable): A function to predicate a node.

    Returns:
        onnx.ModelProto: Modified onnx model.
    """
    # ! this doesn't handle inputs/outputs
    logger = get_root_logger()
    while True:
        connect = None
        for i, node in enumerate(model.graph.node):
            if predicate(node):
                assert len(node.input) == 1
                assert len(node.output) == 1
                connect = (node.input[0], node.output[0])
                logger.info(f'remove node {node.name}')
                del model.graph.node[i]
                break
        if not connect:
            break
        src, dst = connect
        for node in model.graph.node:
            for i, input in enumerate(node.input):
                if input == dst:
                    node.input[i] = src
    return model


def is_unused_mark(marks: Iterable[onnx.NodeProto]) -> Callable:
    """Check whether a mark is unused.

    Args:
        marks (Iterable[onnx.NodeProto]): A list of onnx NodeProto.

    Returns:
        Callable: The function to check if a mark node is in `marks`.
    """

    def f(node):
        if node.op_type == 'Mark':
            attr = attribute_to_dict(node.attribute)
            name = attr['func'] + ':' + attr['type']
            if name not in marks:
                return True
        return False

    return f


def is_identity(node: onnx.NodeProto) -> bool:
    """Check if an op is identity."""
    return node.op_type == 'Identity'


def get_new_name(attrs: Dict[str, str],
                 mark_name: str = '',
                 name_map: Optional[Dict[str, str]] = None) -> str:
    """Get new name for a node.

    Args:
        attrs (Dict[str, str]): A dict contains attributes of an ONNX node.
        mark_name (str): The input mark op name. Default is ''.
        name_map (Dict[str, str]): A mapping of node names, defaults to
            `None`.

    Returns:
        str: The new node name.
    """
    if 'name' in attrs:
        new_name = attrs['name']
    else:
        new_name = '_'.join((attrs['func'], attrs['type'], str(attrs['id'])))

    if name_map is not None:
        if new_name in name_map:
            return name_map[new_name]

        if f'{mark_name}:{new_name}' in name_map:
            return name_map[f'{mark_name}:{new_name}']

    return new_name


def rename_value(model: onnx.ModelProto, old_name: str, new_name: str):
    """Rename a node in an ONNX model.

    Args:
        model (onnx.ModelProto): Input onnx model.
        old_name (str): Original node name in the model.
        new_name (str): New node name in the model.
    """
    if old_name == new_name:
        return
    logger = get_root_logger()
    logger.info(f'rename {old_name} -> {new_name}')
    for n in model.graph.node:
        for i, output in enumerate(n.output):
            if output == old_name:
                n.output[i] = new_name
        for i, input in enumerate(n.input):
            if input == old_name:
                n.input[i] = new_name
    for v in model.graph.value_info:
        if v.name == old_name:
            v.name = new_name
    for i, input in enumerate(model.graph.input):
        if input.name == old_name:
            input.name = new_name
    for i, output in enumerate(model.graph.output):
        if output.name == old_name:
            output.name = new_name


def remove_identity(model: onnx.ModelProto):
    """Remove identity node from an ONNX model.

    Args:
        model (onnx.ModelProto): Input onnx model.
    """
    graph = model.graph

    def simplify_inputs():
        connect = None
        logger = get_root_logger()
        for input in graph.input:
            for i, node in enumerate(graph.node):
                if node.op_type == 'Identity' and node.input[0] == input.name:
                    connect = (node.input[0], node.output[0])
                    logger.info(f'remove node {node.name}')
                    del graph.node[i]
                    break
            if connect:
                break
        if not connect:
            return False
        src, dst = connect
        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == dst:
                    node.input[i] = src
        # the input just changed won't be an output
        return True

    def simplify_outputs():
        connect = None
        logger = get_root_logger()
        for output in graph.output:
            for i, node in enumerate(graph.node):
                if node.op_type == 'Identity' and \
                        node.output[0] == output.name:
                    connect = (node.input[0], node.output[0])
                    logger.info(f'remove node {node.name}')
                    del graph.node[i]
                    break
            if connect:
                break
        if not connect:
            return False
        src, dst = connect
        for node in graph.node:
            for i, output_name in enumerate(node.output):
                if output_name == src:
                    node.output[i] = dst
            # the output just renamed may be someone's input
            for i, input_name in enumerate(node.input):
                if input_name == src:
                    node.input[i] = dst
        return True

    while simplify_inputs():
        pass

    while simplify_outputs():
        pass

    remove_nodes(model, is_identity)

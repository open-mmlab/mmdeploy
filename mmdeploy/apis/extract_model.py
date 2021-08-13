import logging
from typing import Dict, Iterable, Optional, Union

import onnx
import onnx.helper
import onnx.utils
import torch.multiprocessing as mp

from mmdeploy.utils import parse_extractor_io_string
from .utils import attribute_to_dict


def _dfs_search_reacable_nodes_fast(self, node_output_name, graph_input_nodes,
                                    reachable_nodes):
    outputs = {}
    for index, node in enumerate(self.graph.node):
        for name in node.output:
            if name not in outputs:
                outputs[name] = set()
            outputs[name].add(index)

    def impl(node_output_name, graph_input_nodes, reachable_nodes):
        if node_output_name in graph_input_nodes:
            return
        if node_output_name not in outputs:
            return
        for index in outputs[node_output_name]:
            node = self.graph.node[index]
            if node in reachable_nodes:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                impl(name, graph_input_nodes, reachable_nodes)

    impl(node_output_name, graph_input_nodes, reachable_nodes)


def remove_nodes(model, predicate):
    # ! this doesn't handle inputs/outputs
    while True:
        connect = None
        for i, node in enumerate(model.graph.node):
            if predicate(node):
                assert len(node.input) == 1
                assert len(node.output) == 1
                connect = (node.input[0], node.output[0])
                logging.info(f'remove node {node.name}')
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


def is_unused_mark(marks):

    def f(node):
        if node.op_type == 'Mark':
            attr = attribute_to_dict(node.attribute)
            name = attr['func'] + ':' + attr['type']
            if name not in marks:
                return True
        return False

    return f


def is_identity(node):
    return node.op_type == 'Identity'


def get_new_name(attrs, mark_name='', name_map=None):
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


def rename_value(model, old_name, new_name):
    if old_name == new_name:
        return
    logging.info(f'rename {old_name} -> {new_name}')
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


def optimize(model):
    graph = model.graph

    def simplify_inputs():
        connect = None
        for input in graph.input:
            for i, node in enumerate(graph.node):
                if node.op_type == 'Identity' and node.input[0] == input.name:
                    connect = (node.input[0], node.output[0])
                    logging.info(f'remove node {node.name}')
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
        for output in graph.output:
            for i, node in enumerate(graph.node):
                if node.op_type == 'Identity' and \
                        node.output[0] == output.name:
                    connect = (node.input[0], node.output[0])
                    logging.info(f'remove node {node.name}')
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


def extract_model(model: Union[str, onnx.ModelProto],
                  start: Union[str, Iterable[str]],
                  end: Union[str, Iterable[str]],
                  start_name_map: Optional[Dict[str, str]] = None,
                  end_name_map: Optional[Dict[str, str]] = None,
                  dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                  save_file: Optional[str] = None,
                  ret_value: Optional[mp.Value] = None):

    # set init flag for multiprocessor
    if ret_value is not None:
        ret_value.value = -1

    if isinstance(model, str):
        model = onnx.load(model)

    num_value_info = len(model.graph.value_info)
    inputs = []
    outputs = []
    if not isinstance(start, (list, tuple)):
        start = [start]
    for s in start:
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

    logging.info(f'inputs: {", ".join(inputs)}')

    # collect outputs
    if not isinstance(end, (list, tuple)):
        end = [end]
    for e in end:
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

    logging.info(f'outputs: {", ".join(outputs)}')

    # replace Mark with Identity
    for node in model.graph.node:
        if node.op_type == 'Mark':
            del node.attribute[:]
            node.domain = ''
            node.op_type = 'Identity'

    # patch extractor
    onnx.utils.Extractor._dfs_search_reachable_nodes = \
        _dfs_search_reacable_nodes_fast

    extractor = onnx.utils.Extractor(model)
    extracted_model = extractor.extract_model(inputs, outputs)

    # remove all Identity, this may be done by onnx simplifier
    optimize(extracted_model)

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
                logging.info(f'fixing output shape: {x.name}')
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

    # set success flag for multiprocessor
    if ret_value is not None:
        ret_value.value = 0

    return extracted_model

import argparse
import os.path as osp

import onnx
import onnx.helper
import onnx.utils
from onnx import AttributeProto


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract model based on markers.')
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument(
        '--start',
        help='Start markers, format: func:type, e.g. backbone:input')
    parser.add_argument('--end', help='End markers')

    args = parser.parse_args()

    args.start = args.start.split(',') if args.start else []
    args.end = args.end.split(',') if args.end else []

    return args


def remove_markers(model):
    shortcut = []
    success = True
    while success:
        success = False
        for i, node in enumerate(model.graph.node):
            if node.op_type == 'Mark':
                for input in node.input:
                    shortcut.append((input, node.output))
                del model.graph.node[i]
                success = True
                break
    for src, dsts in shortcut:
        for curr in model.graph.node:
            for k, input in enumerate(curr.input):
                if input in dsts:
                    curr.input[k] = src
        # TODO: handle duplicated case?
        for k, output in enumerate(model.graph.output):
            print(output.name, dsts)
            if output.name in dsts:
                output.name = src
    return model


def attribute_to_dict(attribute):
    ret = {}
    for a in attribute:
        name = a.name
        if a.type == AttributeProto.AttributeType.STRING:
            ret[name] = str(a.s, 'utf-8')
        elif a.type == AttributeProto.AttributeType.INT:
            ret[name] = a.i
    return ret


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


def get_new_name(attrs):
    if 'name' in attrs:
        return attrs['name']
    return '_'.join((attrs['func'], attrs['type'], str(attrs['id'])))


def rename_value(model, old_name, new_name):
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
    for i, name in enumerate(model.graph.input):
        if name == old_name:
            model.graph.input[i] = new_name
    for i, name in enumerate(model.graph.output):
        if name == old_name:
            model.graph.output[i] = new_name


def extract_model(model, start, end):
    inputs = []
    outputs = []
    if not isinstance(start, (list, tuple)):
        start = [start]
    for s in start:
        start_name, start_type = s.split(':')
        assert start_type in ['input', 'output']
        for node in model.graph.node:
            if node.op_type == 'Mark':
                attr = attribute_to_dict(node.attribute)
                if attr['func'] == start_name and attr['type'] == start_type:
                    name = node.output[
                        0] if start_type == 'input' else node.input[0]
                    if name not in inputs:
                        new_name = get_new_name(attr)
                        rename_value(model, name, new_name)
                        inputs.append(new_name)

    print(f'inputs: {inputs}')

    # collect outputs
    # outputs = []
    if not isinstance(end, (list, tuple)):
        end = [end]
    for e in end:
        end_name, end_type = e.split(':')
        assert end_type in ['input', 'output']
        for node in model.graph.node:
            if node.op_type == 'Mark':
                attr = attribute_to_dict(node.attribute)
                if attr['func'] == end_name and attr['type'] == end_type:
                    name = node.input[
                        0] if end_type == 'input' else node.output[0]
                    if name not in outputs:
                        new_name = get_new_name(attr)
                        rename_value(model, name, new_name)
                        outputs.append(new_name)

    print(f'outputs: {outputs}')

    # replace Mark with Identity
    for node in model.graph.node:
        if node.op_type == 'Mark':
            del node.attribute[:]
            node.domain = ''
            node.op_type = 'Identity'

    # patch extractor
    setattr(onnx.utils.Extractor, '_dfs_search_reachable_nodes',
            _dfs_search_reacable_nodes_fast)

    extractor = onnx.utils.Extractor(model)
    extracted_model = extractor.extract_model(inputs, outputs)

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
                print(f'fixing output shape: {x.name}')
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
    while success:
        success = False
        for i, x in enumerate(extracted_model.graph.value_info):
            if x.name in inputs:
                del extracted_model.graph.value_info[i]
                success = True
                break

    return extracted_model


def collect_avaiable_marks(model):
    marks = []
    for node in model.graph.node:
        if node.op_type == 'Mark':
            attr = attribute_to_dict(node.attribute)
            func = attr['func']
            if func not in marks:
                marks.append(func)
    return marks


def main():
    args = parse_args()

    model = onnx.load(args.input_model)
    marks = collect_avaiable_marks(model)
    print('Available marks:\n    {}'.format('\n    '.join(marks)))

    extracted_model = extract_model(model, args.start, args.end)

    if osp.splitext(args.output_model)[-1] != '.onnx':
        args.output_model += '.onnx'
    onnx.save(extracted_model, args.output_model)


if __name__ == '__main__':
    main()

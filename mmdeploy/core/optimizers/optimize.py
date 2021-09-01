import logging

from onnx.helper import get_attribute_value


def attribute_to_dict(attr):
    ret = {}
    for a in attr:
        value = get_attribute_value(a)
        if isinstance(value, bytes):
            value = str(value, 'utf-8')
        ret[a.name] = value
    return ret


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

# Copyright (c) OpenMMLab. All rights reserved.

import copy

from mmengine import print_log
from onnx import numpy_helper


class ONNXOptimUtils():

    @classmethod
    def map_name_and_data(cls, onnx_model):
        params = {}
        for init in onnx_model.graph.initializer:
            params[init.name] = numpy_helper.to_array(init)
        for node in onnx_model.graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        params[node.output[0]] = numpy_helper.to_array(attr.t)
        return params

    @classmethod
    def map_name_and_initializer(cls, onnx_model, allow_redundant=True):

        initializers = dict()

        for idx, init in enumerate(onnx_model.graph.initializer):
            initializers[init.name] = (init, idx)

        return initializers

    @classmethod
    def map_output_and_node(cls, onnx_model):
        output2node = dict()
        for node in onnx_model.graph.node:
            for output_name in node.output:
                output2node[output_name] = node
        return output2node

    @classmethod
    def map_input_and_node(cls, onnx_model):

        input2node = dict()
        for node in onnx_model.graph.node:
            for idx, input_name in enumerate(node.input):
                if input_name not in input2node:
                    input2node[input_name] = []
                input2node[input_name].append([node, idx])
        return input2node

    @classmethod
    def remove_node_from_onnx(cls, node, onnx_model):
        onnx_model.graph.node.remove(node)

    @classmethod
    def remove_initializer_from_onnx(cls, initializer, onnx_model):
        onnx_model.graph.initializer.remove(initializer)

    @classmethod
    def remove_fake_pad_op(cls, onnx_model, name2data, inp2node, out2node):
        nodes_to_be_removed = []
        for idx, node in enumerate(onnx_model.graph.node):
            if node.op_type == 'Pad':
                pads = name2data[node.input[1]]
                if all([x == 0 for x in pads]):
                    print_log(f'Remove pad op: <{node.name}>.')
                    next_nodes = inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]
                    nodes_to_be_removed.append(node)

        for node in nodes_to_be_removed:
            onnx_model.graph.node.remove(node)

    @classmethod
    def insert_node_to_onnx(cls, node, onnx_model, idx=0):
        onnx_model.graph.node.insert(idx, node)

    @classmethod
    def find_standalone_nodes(cls,
                              onnx_model,
                              input2node=None,
                              output2node=None):

        if input2node is None:
            input2node = cls.map_input_and_node(onnx_model)
        if output2node is None:
            output2node = cls.map_output_and_node(onnx_model)

        def _is_standalone_node(node, input2node, output2node):
            standalone = True
            for input_name in node.input:
                if input_name in output2node:
                    standalone = False
                    break

            if not standalone:
                return False

            for out_node in node.output:
                if out_node in input2node:
                    standalone = False

            return standalone

        standalone_nodes = list()
        for node in onnx_model.graph.node:

            if _is_standalone_node(node, input2node, output2node):
                standalone_nodes.append(node)
        return standalone_nodes

    @classmethod
    def find_redundant_initializers(cls, onnx_model, input2node=None):
        if input2node is None:
            input2node = cls.map_input_and_node(onnx_model)

        initializers = cls.map_name_and_initializer(onnx_model)
        redundant_initializers = list()
        redundant_set = set()
        for name, init_and_idx in initializers.items():
            if name not in input2node and name not in redundant_set:
                redundant_initializers.append(init_and_idx[0])
                redundant_set.add(name)
        return redundant_initializers

    @classmethod
    def topo_sort(cls, onnx_model, initializers=None, inplace=True):

        def _is_zero_in_degree(node, exist_inputs, initializers):
            flag = True
            for input_name in node.input:
                if (input_name not in exist_inputs
                        and input_name not in initializers):
                    flag = False
                    break

            return flag

        if inplace:
            _onnx_model = onnx_model
        else:
            _onnx_model = copy.deepcopy(onnx_model)

        if initializers is None:
            initializers = cls.map_name_and_initializer(
                _onnx_model, allow_redundant=True)

        visited_inputs = [node.name for node in _onnx_model.graph.input]
        num_nodes = len(_onnx_model.graph.node)

        sorted_nodes = list()

        while len(sorted_nodes) < num_nodes:
            find_new_node = False
            for i in range(num_nodes):
                node = _onnx_model.graph.node[i]

                if node.name in sorted_nodes:
                    continue

                if _is_zero_in_degree(node, visited_inputs, initializers):

                    find_new_node = True
                    sorted_nodes.append(node.name)
                    _onnx_model.graph.node.append(node)
                    for output_name in node.output:
                        visited_inputs.append(output_name)

            assert find_new_node, 'Graph is illegel, error occurred!'

        for i in range(num_nodes):
            remove_node = _onnx_model.graph.node[0]
            _onnx_model.graph.node.remove(remove_node)

        return _onnx_model

    @classmethod
    def optimize(cls, onnx_model):

        standalone_nodes = cls.find_standalone_nodes(onnx_model)
        for node in standalone_nodes:
            cls.remove_node_from_onnx(node, onnx_model)
            print_log(f'Remove node {node.name}')

        redundant_inits = cls.find_redundant_initializers(onnx_model)
        for init in redundant_inits:
            cls.remove_initializer_from_onnx(init, onnx_model)
            print_log(f'Remove initializer {init.name}')

        sorted_onnx_model = cls.topo_sort(onnx_model)

        return sorted_onnx_model

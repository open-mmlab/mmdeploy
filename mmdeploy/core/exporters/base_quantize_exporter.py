# Copyright (c) OpenMMLab. All rights reserved.

import onnx
from mmengine import print_log
from onnx import numpy_helper

from ..optimizers import ONNXOptimUtils

SUPPORT_QWEIGHT_NODE = ['Gemm', 'Conv', 'ConvTranspose']

PERCHANNEL_FAKEQUANTIZER = [
    'FakeQuantizeLearnablePerchannelAffine', 'FixedPerChannelAffine',
    'FakeQuantizeDSQPerchannel'
]
PERTENSOR_FAKEQUANTIZER = [
    'LearnablePerTensorAffine', 'FixedPerTensorAffine',
    'FakeQuantizeDSQPertensor', 'FakeQuantizeTqtAffine'
]

ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER


def _parse_attrs(node_attrs):
    attrs = {}
    for attr in node_attrs:
        if attr.type == onnx.AttributeProto.AttributeType.INTS:
            attrs[attr.name] = tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.AttributeType.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            attrs[attr.name] = tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.AttributeType.STRING:
            attrs[attr.name] = str(attr.s)
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            attrs[attr.name] = tuple([str(x) for x in attr.strings])
        else:
            raise Exception('ATTR Type [{}] Not Supported!'.format(attr.type))
    return attrs


class BaseQuantizeExportor():

    optimizer = ONNXOptimUtils

    def __init__(self, onnx_model, export_path) -> None:

        if isinstance(onnx_model, str):
            self.onnx_model = onnx.load(onnx_model)
        elif isinstance(onnx_model, onnx.GraphProto):
            self.onnx_model = onnx_model
        else:
            raise TypeError

        self.export_path = export_path
        self._init_mappings_from_onnx(self.onnx_model)

        self.optimizer.remove_fake_pad_op(self.onnx_model, self.name2data,
                                          self.input2node, self.output2node)

        self._remap_input_and_node()
        self._remap_output_and_node()

    @property
    def graph(self):
        return self.onnx_model.graph

    def _init_mappings_from_onnx(self, onnx_model):

        self.input2node = self.optimizer.map_input_and_node(onnx_model)
        self.output2node = self.optimizer.map_output_and_node(onnx_model)
        self.name2data = self.optimizer.map_name_and_data(onnx_model)
        self.name2init = self.optimizer.map_name_and_initializer(onnx_model)

    def _remap_input_and_node(self):
        self.input2node = self.optimizer.map_input_and_node(self.onnx_model)

    def _remap_output_and_node(self):
        self.output2node = self.optimizer.map_output_and_node(self.onnx_model)

    def parse_qparams(self, node):
        tensor_name, scale, zero_point = node.input[:3]

        scale, zero_point = self.name2data[scale], self.name2data[zero_point]
        if len(node.input) > 3:
            qmin, qmax = node.input[-2:]
            qmin, qmax = self.name2data[qmin], self.name2data[qmax]
        elif len(node.attribute) > 0:
            qparams = _parse_attrs(node.attribute)
            qmin = qparams['quant_min']
            qmax = qparams['quant_max']
        else:
            print_log(f'qmin and qmax are not found for <{node.name}>!')
            qmax = qmin = None
        return tensor_name, scale, zero_point, qmin, qmax

    def collect_symbolic_nodes(self, onnx_model):
        symbolic_nodes = list()
        for node in onnx_model.graph.node:
            if node.op_type in ALL_FAKEQUANTIZER:
                symbolic_nodes.append(node)
        return symbolic_nodes

    def _get_constant_inputs(self, node):
        constant_nodes = list()
        output2node = self.output2node
        for inp in node.input:
            if inp in output2node and output2node[inp].op_type == 'Constant':
                cnode = output2node[inp]

                constant_nodes.append(cnode)
        return constant_nodes

    def _collect_symbolic_constant_inputs(self, symbolic_nodes):

        collected_constant_names = set()
        constant_inputs = list()
        for node in symbolic_nodes:
            constant_inputs = self._get_constant_inputs(node)
            for constant in constant_inputs:
                if constant.name in collected_constant_names:
                    continue
                constant_inputs.append(constant)
                collected_constant_names.add(constant.name)
        return constant_inputs

    def _remove_symbolic_related_from_onnx(self, symbolic_nodes,
                                           symbolic_constant_inputs):
        for node in symbolic_nodes:
            self.onnx_model.graph.node.remove(node)

        for constant in symbolic_constant_inputs:
            remove = True
            for node in self.onnx_model.graph.node:
                for input_name in node.input:
                    if input_name == constant.output[0]:
                        remove = False
                        break
            if remove:
                self.onnx_model.graph.node.remove(constant)

    def export(self, onnx_path):
        pass


class QTableQuantizeExportor(BaseQuantizeExportor):

    def __init__(self, onnx_model, export_path) -> None:
        super().__init__(onnx_model, export_path)

        self._qtables = dict()

    @property
    def qtables(self):
        return self._qtables

    def register_qtables(self, value, key):
        assert value not in self._qtables
        self._qtables[value] = key

    def post_process_qtables(self):

        def find_the_closest_tensor(node):
            if node.input[0] in self.qtables:
                return node.input[0]
            elif (node.op_type in ['Flatten', 'Resize']
                  and node.output[0] in self.input2node):

                next_node = self.input2node[node.output[0]][0][0]
                return find_the_closest_tensor(next_node)
            else:
                return None

        for node in self.graph.node:
            if node.op_type in ['Flatten', 'Resize']:
                tensor_name = find_the_closest_tensor(node)
                if tensor_name:
                    self.qtables[node.input[0]] = self.qtables[tensor_name]
                    print_log(
                        f'Pass <{tensor_name}> clip range to <{node.name}> '
                        f'input <{node.input[0]}>.')

    def _is_fakequant_for_weight(self, node):

        if node.output[0] not in self.input2node:

            assert node.output[0] in [out.name for out in self.graph.output], \
                        f'{node.name} not in graph.'

            self.input2node[node.output[0]] = []
        next_nodes = self.input2node[node.output[0]]

        flag = True
        for n in next_nodes:
            if n[1] == 1 and n[0].op_type in SUPPORT_QWEIGHT_NODE:
                continue
            else:
                flag = False
                break

        return flag

    def _is_fakequant_for_bias(self, node):

        if node.output[0] not in self.input2node:

            assert node.output[0] in [out.name for out in self.graph.output], \
                        f'{node.name} not in graph.'

            self.input2node[node.output[0]] = []
        next_nodes = self.input2node[node.output[0]]

        flag = True
        for n in next_nodes:
            if n[1] == 2 and n[0].op_type in SUPPORT_QWEIGHT_NODE:
                continue
            else:
                flag = False
                break

        return flag

    def _is_fakequant_for_activation(self, node):

        return (not self._is_fakequant_for_weight(node)
                and not self._is_fakequant_for_bias(node))

    def deal_with_weight_fakequant(self, node):

        next_nodes = self.input2node[node.output[0]]
        next_node = next_node, idx = next_nodes[0]
        next_node.input[idx] = node.input[0]

    def deal_with_activation_fakequant(self, node):
        next_nodes = self.input2node[node.output[0]]
        for next_node, idx in next_nodes:
            next_node.input[idx] = node.input[0]

    def deal_with_per_channel_node(self, node):
        # fake quantize for weights
        # suppose per-channel quantize only for weight
        if not self.is_fakequant_for_weight(node):
            raise RuntimeError('Only support per-channel quantize for weight')
        self.deal_with_weight_fakequant(node)

    def deal_with_per_tensor_node(self, node):

        if self._is_fakequant_for_weight(node):
            self.deal_with_per_tensor_weight(node)
        elif self._is_fakequant_for_bias(node):
            self.deal_with_per_tensor_bias(node)
        elif self._is_fakequant_for_activation(node):
            self.deal_with_per_tensor_activation(node)
        else:
            raise NotImplementedError

    def deal_with_per_tensor_weight(self, node):
        # fake quantize for weights
        self.deal_with_weight_fakequant(node)

    def deal_with_per_tensor_bias(self, node):
        # fake quantize for bias
        raise RuntimeError(f"{self.backend} don't support per-tensor quantize "
                           f'for bias')

    def deal_with_per_tensor_activation(self, node):

        # fake quantize for activations

        self.deal_with_activation_fakequant(node)
        tensor_name, _, _, _, _ = self.parse_qparams(node)
        for out in self.graph.output:
            if out.name == node.output[0]:
                out.name = tensor_name

    def _remove_symbolic_and_collect_params(self):
        symbolic_nodes = self.collect_symbolic_nodes(self.onnx_model)

        collect_func = self._collect_symbolic_constant_inputs
        symbolic_constant_inputs = collect_func(symbolic_nodes)

        for node in symbolic_nodes:
            if node.op_type in PERCHANNEL_FAKEQUANTIZER:
                self.deal_with_per_channel_node(node)
            else:
                self.deal_with_per_tensor_node(node)

        self._remove_symbolic_related_from_onnx(symbolic_nodes,
                                                symbolic_constant_inputs)

        self.optimizer.optimize(self.onnx_model)

        self._remap_input_and_node()
        self._remap_output_and_node()

        self.post_process_qtables()

    def export_qtables(self):

        pass

    def export(self):
        self._remove_symbolic_and_collect_params()

        onnx.save(self.onnx_model, self.export_path)

        self.export_qtables()

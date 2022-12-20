# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .base_quantize_exporter import BaseQuantizeExportor


class OpenVinoQuantizeExportor(BaseQuantizeExportor):

    def __init__(self, onnx_model, export_path) -> None:
        super().__init__(onnx_model, export_path)

    def _build_backend_node_from_symbolic(self, node, tensor_name, qmin, qmax):
        qmax = int(qmax)
        qmin = int(qmin)
        levels = qmax - qmin + 1
        # adjust weight levels
        # if levels == 128:
        #     levels = 256
        #     qmax = qmax * 2 + 1
        #     qmin = qmin * 2
        output_name = node.output[0]
        # Create a node (FakeQuantize)
        keys = ['input_min', 'input_max', 'output_min', 'output_max']
        input_names = [f'{tensor_name}_{key}' for key in keys]
        backend_node = helper.make_node(
            'FakeQuantize',  # node name
            [tensor_name, *input_names],  # inputs
            [output_name],  # outputs
            levels=levels,  # Attributes
            domain='org.openvinotoolkit',
            name=node.name)
        return backend_node

    def _build_backend_initializer(self, names, scale, zero_point, qmin, qmax,
                                   shape):

        scale = np.abs(np.asarray(scale, dtype=np.float64).reshape(-1))
        zero_point = np.clip(
            np.asarray(np.round(zero_point), dtype=np.int32).reshape(-1),
            a_min=qmin,
            a_max=qmax)

        qrange = float(qmax - qmin)
        input_range = scale * qrange
        input_high = (qmax - zero_point).astype(
            np.float64) * input_range / qrange
        input_low = input_high - input_range
        input_low_size = input_low.size

        if input_low_size != 1:
            input_low = input_low.reshape(*shape)
            input_high = input_high.reshape(*shape)

        input_low = input_low.astype(np.float32)
        input_high = input_high.astype(np.float32)

        initializers = list()
        for init_name, value_tensor in zip(
                names, [input_low, input_high, input_low, input_high]):
            init = numpy_helper.from_array(value_tensor)
            init.name = init_name
            initializers.append(init)
        return initializers

    def build_backend_nodes_and_initializers(self, symbolic_nodes):
        backend_nodes = list()
        backend_initializers = list()
        for node in symbolic_nodes:
            tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(
                node)
            new_node = self._build_backend_node_from_symbolic(
                node, tensor_name, qmin, qmax)
            backend_nodes.append(new_node)

            try:
                next_node = self.input2node[node.output[0]][0][0]
                # node for save weights
                fake_node = self.output2node[next_node.input[1]]
                tensor = self.name2data[fake_node.input[0]]
                shape_length = len(tensor.shape)
                new_shape = [
                    -1,
                ] + [
                    1,
                ] * (
                    shape_length - 1)
            except Exception:
                new_shape = [
                    -1,
                ]

            new_init_names = new_node.input[1:]
            new_initializers = self._build_backend_initializer(
                new_init_names, scale, zero_point, qmin, qmax, new_shape)
            backend_initializers.extend(new_initializers)
        return backend_nodes, backend_initializers

    def _insert_initializers_to_onnx(self, initializers):
        inserted_init_names = set()
        for init in initializers:
            if init.name in inserted_init_names:
                continue

            self.onnx_model.graph.initializer.append(init)
            inserted_init_names.add(init.name)

    def _replace_symbolic_related(self):

        symbolic_nodes = self.collect_symbolic_nodes(self.onnx_model)

        collect_func = self._collect_symbolic_constant_inputs
        symbolic_constant_inputs = collect_func(symbolic_nodes)

        build_func = self.build_backend_nodes_and_initializers
        new_nodes, new_initializers = build_func(symbolic_nodes)

        self._insert_initializers_to_onnx(new_initializers)

        self._remove_symbolic_related_from_onnx(symbolic_nodes,
                                                symbolic_constant_inputs)

        self.onnx_model.graph.node.extend(new_nodes)
        self.optimizer.optimize(self.onnx_model)

    def export(self):
        self._replace_symbolic_related()
        onnx.save(self.onnx_model, self.export_path)

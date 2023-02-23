# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.fileio import dump
import onnx
from .base_quantize_exporter import QTableQuantizeExportor, BaseQuantizeExportor
import numpy as np

class TensorRTQTableExporter(QTableQuantizeExportor):

    def __init__(self, onnx_model, export_path) -> None:
        super().__init__(onnx_model, export_path)

    def deal_with_per_tensor_activation(self, node):
        super().deal_with_per_tensor_activation(node)

        name, scale, _, qmin, qmax = self.parse_qparams(node)
        self.qtables[name] = float(scale * max(-qmin, qmax))

    def export_qtables(self):
        context = {'tensorrt': {'blob_range': self.qtables}}
        qtables_path = self.export_path.replace('.onnx', '_qtables.json')

        dump(context, qtables_path, 'json')

class TensorRTExplicitExporter(BaseQuantizeExportor):
    
    def __init__(self, onnx_model, export_path) -> None:
        super().__init__(onnx_model, export_path)
    
    def _build_backend_node_from_symbolic(self, node):
        quantize_linear_node = onnx.helper.make_node("QuantizeLinear", 
                                                     node.input[:3],
                                                     [node.name + '_quantized_out'], 
                                                     node.name + '_quantized')
        dequantize_linear_node = onnx.helper.make_node("DequantizeLinear",
                                                        [node.name + '_quantized_out'] + quantize_linear_node.input[1:3],
                                                        node.output,
                                                        node.name + '_dequantized')
        return [quantize_linear_node, dequantize_linear_node]
                
    def build_backend_nodes(self, symbolic_nodes):
        backend_nodes = list()
        for node in symbolic_nodes:
            _, _, zero_point, qmin, qmax = self.parse_qparams(node)
            assert qmax - qmin in (2 ** 8 - 1, 2 ** 8 - 2), "Only 8 bit quantization support deployment to ONNX."
            assert not np.any(zero_point != 0), "This pass is only supposed to be used with TensorRT Backend which" \
                        "does not support asymmetric quantization."
            new_nodes = self._build_backend_node_from_symbolic(node)
            backend_nodes.extend(new_nodes)
        return backend_nodes

    def export(self):
        symbolic_nodes = self.collect_symbolic_nodes(self.onnx_model)
        new_nodes = self.build_backend_nodes(symbolic_nodes)
        for node in symbolic_nodes:
            self.onnx_model.graph.node.remove(node)
        self.onnx_model.graph.node.extend(new_nodes)        
        self.optimizer.optimize(self.onnx_model)
        onnx.save(self.onnx_model, self.export_path)
    
            
            
            
        
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.fileio import dump

from .base_quantize_exporter import QTableQuantizeExportor


class TensorRTQuantizeExporter(QTableQuantizeExportor):

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

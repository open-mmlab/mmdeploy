# Copyright (c) OpenMMLab. All rights reserved.
from .openvino_quantize_exporter import OpenVinoQuantizeExportor
from .tensorrt_quantize_exporter import TensorRTQuantizeExporter

__all__ = ['OpenVinoQuantizeExportor', 'TensorRTQuantizeExporter']

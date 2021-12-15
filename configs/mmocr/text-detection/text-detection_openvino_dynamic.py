_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/openvino.py']

onnx_config = dict(strip_doc_string=False)

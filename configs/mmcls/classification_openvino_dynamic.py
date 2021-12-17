_base_ = ['./classification_static.py', '../_base_/backends/openvino.py']

onnx_config = dict(strip_doc_string=False)

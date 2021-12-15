_base_ = ['./text-detection_static.py', '../../_base_/backends/openvino.py']

onnx_config = dict(input_shape=(640, 640), strip_doc_string=False)

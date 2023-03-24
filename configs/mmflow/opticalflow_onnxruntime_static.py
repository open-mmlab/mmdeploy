_base_ = ['./opticalflow_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(verbose=True)

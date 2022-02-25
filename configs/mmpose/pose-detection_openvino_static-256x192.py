_base_ = ['./pose-detection_static.py', '../_base_/backends/openvino.py']

onnx_config = dict(input_shape=[192, 256])
backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 256, 192]))])

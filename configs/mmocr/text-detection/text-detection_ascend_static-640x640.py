_base_ = ['./text-detection_static.py', '../../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[640, 640])
backend_config = dict(
    model_inputs=[dict(input_shapes=dict(input=[1, 3, 640, 640]))])

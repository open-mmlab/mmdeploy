_base_ = ['./segmentation_static.py', '../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[1024, 512])
backend_config = dict(
    model_inputs=[dict(input_shapes=dict(input=[1, 3, 512, 1024]))])

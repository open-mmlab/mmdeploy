_base_ = ['./segmentation_static.py', '../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[2048, 1024])
backend_config = dict(
    model_inputs=[dict(input_shapes=dict(input=[1, 3, 1024, 2048]))])

_base_ = ['./segmentation_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[320, 320])

codebase_config = dict(with_argmax=False)

backend_config = dict(
    input_size_list=[[3, 320, 320]],
    quantization_config=dict(do_quantization=False))

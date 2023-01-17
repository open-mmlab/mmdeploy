_base_ = ['./segmentation_static.py', '../_base_/backends/ncnn.py']

onnx_config = dict(input_shape=[512, 512])

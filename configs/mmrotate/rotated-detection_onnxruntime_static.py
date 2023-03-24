_base_ = ['./rotated-detection_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(output_names=['dets', 'labels'], input_shape=[1024, 1024])

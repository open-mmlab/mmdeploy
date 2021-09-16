_base_ = ['./base_static.py']

onnx_config = dict(output_names=['dets', 'labels', 'masks'])

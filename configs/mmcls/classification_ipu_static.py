_base_ = ['./classification_static.py', '../_base_/backends/ipu.py']

backend_config = dict(input_shape='input=1,3,224,224')

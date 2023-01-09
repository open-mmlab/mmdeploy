_base_ = ['./segmentation_static.py', '../_base_/backends/ipu.py']


backend_config = dict(input_shape='input=1,3,1024,512',
                      available_memory_proportion=0.2)

onnx_config = dict(input_shape=[512, 1024])

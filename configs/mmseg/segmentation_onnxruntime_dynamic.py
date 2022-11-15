_base_ = ['./segmentation_dynamic.py', '../_base_/backends/onnxruntime.py']
codebase_config = dict(do_argmax=True)

_base_ = ['./segmentation_dynamic.py', '../_base_/backends/onnxruntime.py']

codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=False)
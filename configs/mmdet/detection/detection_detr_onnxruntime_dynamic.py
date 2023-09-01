_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/onnxruntime.py']
onnx_config = dict(
    input_names=['input', 'shape'],
    dynamic_axes={
        'shape': {
            0: 'batch'
        },
    },
)

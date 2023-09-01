_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/tensorrt.py']
onnx_config = dict(
    input_names=['input', 'shape'],
    dynamic_axes={
        'shape': {
            0: 'batch',
        },
    },
)

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                shape=dict(
                    min_shape=[1, 2], opt_shape=[1, 2], max_shape=[2, 2]),
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[2, 3, 1344, 1344])))
    ])

_base_ = ['../_base_/base_static.py', '../../_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=(320, 320))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 320, 320],
                    max_shape=[1, 3, 320, 320])))
    ])

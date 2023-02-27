_base_ = ['../_base_/base_static.py', '../../_base_/backends/vacc.py']

onnx_config = dict(input_shape=[416, 416])

backend_config = dict(model_inputs=[
    dict(shape=dict(input=[1, 3, 416, 416]), qconfig=dict(dtype='int8'))
])

partition_config = dict(
    type='vacc_det',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'],
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])

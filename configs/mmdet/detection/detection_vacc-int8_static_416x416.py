_base_ = ['../_base_/base_static.py', '../../_base_/backends/vacc.py']

backend_config = dict(
    common_config=dict(
        name='yolov3'
    ),
    model_inputs=[
        dict(
            shape=dict(input=[1, 3, 416, 416]),
            qconfig=dict(
                dtype='int8'
            )
        )
    ]
)


partition_config = dict(
    type='vacc_det',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='yolov3.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'],
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])
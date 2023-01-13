_base_ = ['./segmentation_static.py', '../_base_/backends/vacc.py']

backend_config = dict(
    common_config=dict(
        name='fcn'
    ),
    model_inputs=[
        dict(
            shape=dict(input=[1, 3, 512, 512]),
            qconfig=dict(
                dtype='fp16'
            )
        )
    ]
)

partition_config = dict(
    type='end2end',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='fcn.onnx',
            start=['segmentor_forward:output'],
            end=['seg_maps:input'],
            output_names=['feat'])
    ])
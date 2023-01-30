_base_ = ['./classification_static.py', '../_base_/backends/vacc.py']

backend_config = dict(
    common_config=dict(
        model_info='/path/to/model_info.json',
        vdsp_params_info='/path/to/vdsp_params_info.json'
    ),
    model_inputs=[
        dict(
            shape=dict(input=[1, 3, 224, 224]),
            qconfig=dict(
                dtype='fp16'
            )
        )
    ]
)

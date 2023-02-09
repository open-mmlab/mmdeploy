backend_config = dict(
    type='vacc',
    common_config=dict(name='end2end'),
    model_inputs=[
        dict(
            shape=dict(input=[1, 3, 224, 224]),
            qconfig=dict(
                dtype='fp16',
                calibrate_mode='percentile',
                weight_scale='max',
                data_transmode=1,
                per_channel=False,
                cluster_mode=0,
                skip_conv_layers=[],
                calib_num=1000,
            ))
    ])

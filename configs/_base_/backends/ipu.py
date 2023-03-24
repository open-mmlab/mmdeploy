backend_config = dict(
    type='ipu',
    # precision='fp16',
    precision='fp32',
    output_dir='',
    batches_per_step=1,
    ipu_version='ipu21',
    popart_options=dict(
        rearrangeAnchorsOnHost='false',
        enablePrefetchDatastreams='false',
        groupHostSync='false',
        # partialsTypeMatMuls='half'
    ))

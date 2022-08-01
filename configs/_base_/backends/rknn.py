backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=[123.675, 116.28, 103.53],
        std_values=[58.395, 57.12, 57.375],
        output_tensor_type=None,
        target_platform='rk3588',
        optimization_level=3),
    quantization_config=dict(do_quantization=True, dataset=None))

backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=None,  # [[123.675, 116.28, 103.53]],
        std_values=None,  # [[58.395, 57.12, 57.375]],
        target_platform='rv1126',  # 'rk3588'
        optimization_level=1),
    quantization_config=dict(do_quantization=False, dataset=None))

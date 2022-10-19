backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=None,  # [[103.53, 116.28, 123.675]],
        std_values=None,  # [[57.375, 57.12, 58.395]],
        target_platform='rv1126',  # 'rk3588'
        optimization_level=1),
    quantization_config=dict(do_quantization=False, dataset=None))

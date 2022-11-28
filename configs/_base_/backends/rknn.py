backend_config = dict(
    type='rknn',
    common_config=dict(
        target_platform='rv1126',  # 'rk3588'
        optimization_level=1),
    quantization_config=dict(do_quantization=True, dataset=None))

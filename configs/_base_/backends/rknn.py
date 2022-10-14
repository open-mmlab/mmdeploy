backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=None,
        std_values=None,
        target_platform='rv1126',
        optimization_level=3),
    quantization_config=dict(do_quantization=False, dataset=None))

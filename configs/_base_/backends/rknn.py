backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=None,
        std_values=None,
        output_tensor_type=None,
        target_platform='rk3588',
        optimization_level=3),
    quantization_config=dict(do_quantization=False, dataset=None))

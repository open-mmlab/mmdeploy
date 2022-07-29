backend_config = dict(
    type='rknn',
    common_config=dict(
        mean_values=[123.675, 116.28, 103.53],
        std_values=[58.395, 57.12, 57.375],
        output_tensor_type=None,
        quantized_algorithm='normal',
        target_platform='rk3588',
        float_dtype='float16',
        optimization_level=0),
    quantization_config=dict(do_quantization=False, dataset=None))

backend_config = dict(
    type='coreml',
    # mlprogram or neuralnetwork
    convert_to='mlprogram',
    common_config=dict(
        # FLOAT16 or FLOAT32, see coremltools.precision
        compute_precision='FLOAT32',
        # iOS15, iOS16, etc., see coremltools.target
        minimum_deployment_target='iOS16',
        skip_model_load=False),
)

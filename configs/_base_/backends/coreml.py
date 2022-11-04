backend_config = dict(
    type='coreml',
    convert_to='mlprogram',  # mlprogram or neuralnetwork
    common_config=dict(
        compute_precision='FLOAT32',  # FLOAT16 or FLOAT32, see coremltools.precision
        minimum_deployment_target='iOS16',  # iOS15, iOS16, etc., see coremltools.target
        skip_model_load=False
    ),
)

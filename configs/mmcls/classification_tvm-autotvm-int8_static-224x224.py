_base_ = ['./classification_tvm-autotvm_static-224x224.py']

calib_config = dict(create_calib=True, calib_file='calib_data.h5')
backend_config = dict(model_inputs=[
    dict(
        shape=dict(input=[1, 3, 224, 224]),
        dtype=dict(input='float32'),
        tuner=dict(
            type='AutoTVMTuner',
            log_file='tvm_tune_log.log',
            n_trial=1000,
            tuner=dict(type='XGBTuner'),
        ),
        qconfig=dict(calibrate_mode='kl_divergence', weight_scale='max'),
    )
])

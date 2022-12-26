_base_ = ['../_base_/base_static.py', '../../_base_/backends/tvm.py']

onnx_config = dict(input_shape=[300, 300])
backend_config = dict(model_inputs=[
    dict(
        use_vm=True,
        shape=dict(input=[1, 3, 300, 300]),
        dtype=dict(input='float32'),
        tuner=dict(
            type='AutoTVMTuner',
            log_file='tvm_tune_log.log',
            n_trial=1000,
            tuner=dict(type='XGBTuner'),
        ))
])

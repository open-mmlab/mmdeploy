_base_ = ['./segmentation_static.py', '../_base_/backends/tvm.py']

onnx_config = dict(input_shape=[1024, 512])
backend_config = dict(model_inputs=[
    dict(
        shape=dict(input=[1, 3, 512, 1024]),
        dtype=dict(input='float32'),
        tuner=dict(
            type='AutoTVMTuner',
            log_file='tvm_tune_log.log',
            n_trial=1000,
            tuner=dict(type='XGBTuner')))
])

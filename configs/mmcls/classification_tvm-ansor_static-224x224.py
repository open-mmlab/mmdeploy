_base_ = ['./classification_static.py', '../_base_/backends/tvm.py']

onnx_config = dict(input_shape=[224, 224])
backend_config = dict(model_inputs=[
    dict(
        shape=dict(input=[1, 3, 224, 224]),
        dtype=dict(input='float32'),
        tuner=dict(
            type='AutoScheduleTuner',
            log_file='tvm_tune_log.log',
            num_measure_trials=2000))
])

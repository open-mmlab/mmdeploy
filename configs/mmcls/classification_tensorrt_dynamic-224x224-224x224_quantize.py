_base_ = ['./classification_dynamic.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=[224, 224])
backend_config = dict(
    common_config=dict(
        max_workspace_size=1 << 30,
        explicit_quant_mode=True),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 224, 224],
                    opt_shape=[4, 3, 224, 224],
                    max_shape=[8, 3, 224, 224])))
    ])

global_qconfig=dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8',
        bit=8,
        is_symmetry=True,
        is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8',
        bit=8,
        is_symmetry=True,
        averaging_constant=0.1),
)

quantizer=dict(
    type='mmrazor.TensorRTQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ]
    )
)

checkpoint='/nvme/humu/experiments/ptq_openvino_r50/model_ptq_deploy.pth'
_base_ = ['./classification_dynamic.py', '../_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 224, 224]))])

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
    type='mmrazor.OpenVINOQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ]
    )
)

checkpoint='/mnt/petrelfs/humu/mmrazor/work_dirs/ptq_openvino_resnet18_8xb32_in1k_calib32xb32/model_ptq_deploy.pth'
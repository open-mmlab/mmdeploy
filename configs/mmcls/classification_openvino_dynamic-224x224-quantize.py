_base_ = ['./classification_dynamic.py', '../_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 224, 224]))])

# TODO save in mmrazor's checkpoint
quantizer = dict(
    type='mmrazor.TensorRTQuantizer',
    is_qat=True,
    skipped_methods=[
        'mmcls.models.heads.ClsHead._get_loss',
        'mmcls.models.heads.ClsHead._get_predictions'
    ],
    prepare_custom_config_dict=None,
    convert_custom_config_dict=None,
    qconfig=dict(
        qtype='affine',
        w_observer=dict(type='mmrazor.MinMaxObserver'),
        a_observer=dict(type='mmrazor.EMAMinMaxObserver'),
        w_fake_quant=dict(type='mmrazor.FakeQuantize'),
        a_fake_quant=dict(type='mmrazor.FakeQuantize'),
        w_qscheme=dict(
            bit=8,
            is_symmetry=True,
            is_per_channel=True,
            is_pot_scale=False,
        ),
        a_qscheme=dict(
            bit=8, is_symmetry=False, is_per_channel=False,
            is_pot_scale=False)))

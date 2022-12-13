_base_ = ['../_base_/base_openvino_dynamic-800x1344.py']

# TODO save in mmrazor's checkpoint
quantizer = dict(
    type='mmrazor.OpenvinoQuantizer',
    skipped_methods=[
        'mmdet.models.dense_heads.base_dense_head'
        '.BaseDenseHead.predict_by_feat',
    ],
    prepare_custom_config_dict=None,
    convert_custom_config_dict=None,
    qconfig=dict(
        qtype='affine',
        w_observer=dict(type='mmrazor.MSEObserver'),
        a_observer=dict(type='mmrazor.EMAMSEObserver'),
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

# Copyright (c) OpenMMLab. All rights reserved.
_base_ = 'model.py'

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model={{_base_.model}},
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher={{_base_.model}},
        teacher_trainable=False,
        components=[
            dict(
                student_module='decode_head.conv_seg',
                teacher_module='decode_head.conv_seg',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_logits',
                        tau=1,
                        loss_weight=5,
                    )
                ])
        ]),
)

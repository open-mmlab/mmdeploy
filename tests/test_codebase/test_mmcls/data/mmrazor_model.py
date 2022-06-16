# Copyright (c) OpenMMLab. All rights reserved.
_base_ = 'model.py'

norm_cfg = dict(type='BN')

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
                shuffle_5x5=dict(
                    type='ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
                shuffle_7x7=dict(
                    type='ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
                shuffle_xception=dict(
                    type='ShuffleXception', norm_cfg=norm_cfg),
            ))))

algorithm = dict(
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model={{_base_.model}},
    ),
    mutator=mutator,
    distiller=None,
    mutable_cfg='tests/test_codebase/test_mmcls/data/mmrazor_mutable_cfg.yaml',
    retraining=True)

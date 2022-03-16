# Copyright (c) OpenMMLab. All rights reserved.
_base_ = 'model.py'

norm_cfg = dict(type='BN', requires_grad=True)
mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
                shuffle_5x5=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
                shuffle_7x7=dict(
                    type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
                shuffle_xception=dict(
                    type='ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ))))

algorithm = dict(
    type='DetNAS',
    architecture=dict(
        type='MMDetArchitecture',
        model={{_base_.model}},
    ),
    mutator=mutator,
    pruner=None,
    distiller=None,
    retraining=True,
    mutable_cfg='tests/test_codebase/test_mmdet/data/mmrazor_mutable_cfg.yaml',
)

# Copyright (c) OpenMMLab. All rights reserved.
MMDET_PARTITION_CFG = dict(
    single_stage=[
        dict(
            save_file='partition0.onnx',
            start='detector_forward:input',
            end='multiclass_nms:input',
            dynamic_axes={
                'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                },
                'scores': {
                    0: 'batch',
                    1: 'num_boxes',
                },
                'boxes': {
                    0: 'batch',
                    1: 'num_boxes',
                },
            },
        )
    ],
    two_stage=[
        dict(
            save_file='partition0.onnx',
            start='detector_forward:input',
            end=['extract_feat:output', 'multiclass_nms[0]:input'],
            dynamic_axes={
                'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                },
                'scores': {
                    0: 'batch',
                    1: 'num_boxes',
                },
                'boxes': {
                    0: 'batch',
                    1: 'num_boxes',
                },
            },
        ),
        dict(
            save_file='partition1.onnx',
            start='roi_extractor:output',
            end='bbox_head_forward:output',
            dynamic_axes={
                'bbox_feats': {
                    0: 'batch'
                },
                'cls_score': {
                    0: 'batch'
                },
                'bbox_pred': {
                    0: 'batch'
                },
            },
        )
    ])

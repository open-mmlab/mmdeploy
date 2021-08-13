MMDET_SPLIT_CFG = dict(
    single_stage_base=[
        dict(
            save_file='split0.onnx',
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
    two_stage_base=[
        dict(
            save_file='split0.onnx',
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
            save_file='split1.onnx',
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


def get_split_cfg(split_type):
    assert (split_type in MMDET_SPLIT_CFG), f'Unknow split_type {split_type}'
    return MMDET_SPLIT_CFG[split_type]

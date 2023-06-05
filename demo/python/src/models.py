# Copyright (c) OpenMMLab. All rights reserved.
MODEL_DRROR = -1
UNKNOWN_MODEL = 0
DEEPPOSE_RESNET_50 = 1
DEEPPOSE_RESNET_152 = 2
DEEPPOSE_RESNET_152_RLE = 3

model_dict = dict({
    'deeppose_resnet_50': DEEPPOSE_RESNET_50,
    'deeppose_resnet_152': DEEPPOSE_RESNET_152,
    'deeppose_resnet_152_rle': DEEPPOSE_RESNET_152_RLE
})


def check_model(path: str):
    s = path[:]

    if s.endswith('/'):
        s = s[:-1]

    for t in model_dict:
        if s.endswith(t):
            return model_dict[t]

    return UNKNOWN_MODEL

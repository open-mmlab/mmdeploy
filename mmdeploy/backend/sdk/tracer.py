# Copyright (c) OpenMMLab. All rights reserved.

import json
from hashlib import sha256
from typing import Dict, List, Tuple

from mmcv.utils import Registry


def __build_tracer_wrapper_func(name: str, registry: Registry):
    return registry.module_dict[name]


_TRANSFORM_WRAPPER = Registry('_TRANSFORM', __build_tracer_wrapper_func)


class State:
    """Image info."""

    def __init__(self):
        self.dtype = None
        self.color_type = None


@_TRANSFORM_WRAPPER.register_module(name='LoadImageFromFile')
def load(int_state: State, cur_state: State, args: Dict, transforms: List):
    default_args = {'to_float32': False, 'color_type': 'color'}

    color_type = args.get('color_type', default_args['color_type'])
    if color_type == 'color' or \
            color_type == 'color_ignore_orientation':
        transforms.append({'type': 'cvtColorBGR'})
        int_state.color_type = 'BGR'
        cur_state.color_type = 'BGR'
    else:
        transforms.append({'type': 'cvtColorGray'})
        int_state.color_type = 'GRAY'
        cur_state.color_type = 'GRAY'

    to_float32 = args.get('to_float32', default_args['to_float32'])
    if to_float32 is True:
        transforms.append({'type': 'CastFloat'})
        int_state.dtype = 'float32'
        cur_state.dtype = 'float32'

    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='DefaultFormatBundle')
def default_format_bundle(int_state: State, cur_state: State, args: Dict,
                          transforms: List):
    default_args = {'img_to_float': True}

    img_to_float = args.get('img_to_float', default_args['img_to_float'])
    if img_to_float and (int_state.dtype is None
                         or int_state.dtype != 'float32'):
        transforms.append({'type': 'CastFloat'})
        int_state.dtype = 'float32'
        cur_state.dtype = 'float32'

    transforms.append({'type': 'HWC2CHW'})
    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='Resize')
def resize(int_state: State, cur_state: State, args: Dict, transforms: List):
    transforms.append({'type': 'Resize'})
    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='CenterCrop')
def center_crop(int_state: State, cur_state: State, args: Dict,
                transforms: List):
    transforms.append({'type': 'CenterCrop'})
    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='Normalize')
def normalize(int_state: State, cur_state: State, args: Dict,
              transforms: List):
    default_args = {'to_rgb': True}

    if int_state.dtype is None or int_state.dtype != 'float32':
        transforms.append({'type': 'CastFloat'})
        int_state.dtype = 'float32'
        cur_state.dtype = 'float32'

    to_rgb = args.get('to_rgb', default_args['to_rgb'])
    if to_rgb is True:
        transforms.append({'type': 'cvtColorRGB'})
        cur_state.color_type = 'RGB'

    transforms.append({'type': 'Normalize'})

    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='ImageToTensor')
def image_to_tensor(int_state: State, cur_state: State, args: Dict,
                    transforms: List):
    transforms.append({'type': 'HWC2CHW'})
    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='Collect')
def collect(int_state: State, cur_state: State, args: Dict, transforms: List):
    return True, int_state, cur_state, transforms


@_TRANSFORM_WRAPPER.register_module(name='Pad')
def pad(int_state: State, cur_state: State, args: Dict, transforms: List):
    if int_state.dtype != 'float32':
        return False, int_state, cur_state, transforms

    transforms.append({'type': 'Pad'})
    return True, int_state, cur_state, transforms


def add_transform_tag(pipeline_info: Dict, tag: str) -> Dict:
    if tag is None:
        return pipeline_info

    pipeline_info['pipeline']['tasks'][0]['sha256'] = tag
    pipeline_info['pipeline']['tasks'][0]['fuse_transform'] = False
    return pipeline_info


def get_transform_static(transforms: List) -> Tuple:
    """Get the static transform information for Elena use.

    Args:
        transforms (List): transforms in model_cfg

    Return:
        tuple(): Composed of the static transform information and the tag.
    """

    # Current only support basic transform
    supported_type = [
        'LoadImageFromFile', 'DefaultFormatBundle', 'Resize', 'CenterCrop',
        'Normalize', 'ImageToTensor', 'Collect', 'Pad'
    ]

    # each transform can only appear once
    cnt = {}
    for trans in transforms:
        tp = trans['type']
        if tp not in supported_type:
            return None, None
        if tp in cnt:
            return None, None
        cnt[tp] = 1

    int_state = State()
    cur_state = State()
    elena_transforms = []
    for trans in transforms:
        tp = trans['type']
        args = trans
        func = _TRANSFORM_WRAPPER.build(tp)
        flag, int_state, cur_state, elena_transforms = func(
            int_state, cur_state, args, elena_transforms)
        if flag is False:
            return None, None

    if int_state.dtype != 'float32':
        return None, None

    tag = sha256(json.dumps(elena_transforms).encode('utf-8')).hexdigest()
    return elena_transforms, tag

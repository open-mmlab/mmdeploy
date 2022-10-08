# Copyright (c) OpenMMLab. All rights reserved.

import json
from hashlib import sha256
from typing import Dict, List, Tuple


class TraceFunc:
    """Trace Transform."""

    def __init__(self):
        self.module_dict = dict()

    def register_module(self, name):
        if name in self.module_dict:
            raise KeyError(f'{name} is already registered')

        def _register(func):
            self.module_dict[name] = func
            return func

        return _register

    def get(self, name):
        return self.module_dict[name]


_TRANSFORM_WRAPPER = TraceFunc()


class Context:
    """Trace Context."""

    def __init__(self):
        self.dtype = None
        self.transforms = []


@_TRANSFORM_WRAPPER.register_module(name='LoadImageFromFile')
def load(context: Context, args: Dict):
    default_args = {'to_float32': False, 'color_type': 'color'}
    color_type = args.get('color_type', default_args['color_type'])
    if color_type == 'color' or \
            color_type == 'color_ignore_orientation':
        context.transforms.append({'type': 'cvtColorBGR'})
    else:
        context.transforms.append({'type': 'cvtColorGray'})
    to_float32 = args.get('to_float32', default_args['to_float32'])
    if to_float32 is True:
        context.transforms.append({'type': 'CastFloat'})
        context.dtype = 'float32'
    return True


@_TRANSFORM_WRAPPER.register_module(name='DefaultFormatBundle')
def default_format_bundle(context: Context, args: Dict):
    default_args = {'img_to_float': True}
    img_to_float = args.get('img_to_float', default_args['img_to_float'])
    if img_to_float and (context.dtype is None or context.dtype != 'float32'):
        context.transforms.append({'type': 'CastFloat'})
        context.dtype = 'float32'
    context.transforms.append({'type': 'HWC2CHW'})
    return True


@_TRANSFORM_WRAPPER.register_module(name='Resize')
def resize(context: Context, args: Dict):
    context.transforms.append({'type': 'Resize'})
    return True


@_TRANSFORM_WRAPPER.register_module(name='CenterCrop')
def center_crop(context: Context, args: Dict):
    context.transforms.append({'type': 'CenterCrop'})
    return True


@_TRANSFORM_WRAPPER.register_module(name='Normalize')
def normalize(context: Context, args: Dict):
    default_args = {'to_rgb': True}
    if context.dtype is None or context.dtype != 'float32':
        context.transforms.append({'type': 'CastFloat'})
        context.dtype = 'float32'
    to_rgb = args.get('to_rgb', default_args['to_rgb'])
    if to_rgb is True:
        context.transforms.append({'type': 'cvtColorRGB'})
    context.transforms.append({'type': 'Normalize'})
    return True


@_TRANSFORM_WRAPPER.register_module(name='ImageToTensor')
def image_to_tensor(context: Context, args: Dict):
    context.transforms.append({'type': 'HWC2CHW'})
    return True


@_TRANSFORM_WRAPPER.register_module(name='Pad')
def pad(context: Context, args: Dict):
    if context.dtype != 'float32':
        return False
    context.transforms.append({'type': 'Pad'})
    return True


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

    context = Context()
    for trans in transforms:
        tp = trans['type']
        if tp == 'Collect':
            continue
        args = trans
        func = _TRANSFORM_WRAPPER.get(tp)
        if func(context, args) is False:
            return None, None

    if context.dtype != 'float32':
        return None, None

    tag = sha256(json.dumps(context.transforms).encode('utf-8')).hexdigest()
    return context.transforms, tag

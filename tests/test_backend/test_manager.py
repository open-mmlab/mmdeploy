# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import pytest

from mmdeploy.backend.base import BaseBackendParam


def test_parse_shape():

    parser = argparse.ArgumentParser()
    BaseBackendParam.add_arguments(parser)

    args = parser.parse_args(['--input-shapes', 'input:1x2x3x4'])
    assert args.input_shapes == dict(input=[1, 2, 3, 4])

    args = parser.parse_args(
        ['--input-shapes', 'input1:1x2x3x4,input2:4x3x2x1'])
    assert args.input_shapes == dict(input1=[1, 2, 3, 4], input2=[4, 3, 2, 1])

    args = parser.parse_args(
        ['--input-shapes', 'input1:1x2x3x4', 'input2:4x3x2x1'])
    assert args.input_shapes == dict(input1=[1, 2, 3, 4], input2=[4, 3, 2, 1])

    args = parser.parse_args(
        ['--min-shapes', 'input1:1x2x3x4, input2:4x3x?x?'])
    assert args.min_shapes == dict(
        input1=[1, 2, 3, 4], input2=[4, 3, None, None])

    # input placeholder
    with pytest.raises(ValueError):
        parser.parse_args(['--input-shapes', 'input1:1x2x3x4, input2:4x3x?x?'])

    # duplicate assign
    with pytest.raises(NameError):
        parser.parse_args(['--input-shapes', 'input1:1x2x3x4, input1:4x3x2x1'])

    args = parser.parse_args(['--input-shapes', '1x2x3x4'])
    assert args.input_shapes == {None: [1, 2, 3, 4]}


def test_fix_param():

    params = BaseBackendParam()
    params.input_names = ['input']
    params.input_shapes = {None: [1, 2, 3, 4]}
    params.min_shapes = {None: [1, 2, 3, 4]}
    params.max_shapes = {None: [1, 2, 3, 4]}

    params.fix_param()
    assert params.input_shapes == {'input': [1, 2, 3, 4]}
    assert params.min_shapes == {'input': [1, 2, 3, 4]}
    assert params.max_shapes == {'input': [1, 2, 3, 4]}

    params.min_shapes = {None: [1, 2, None, None]}
    params.max_shapes = {None: [1, 2, None, None]}
    params.fix_param()
    assert params.min_shapes == {'input': [1, 2, 3, 4]}
    assert params.max_shapes == {'input': [1, 2, 3, 4]}

    params = BaseBackendParam(input_shapes={'input': [1, 2, 3, 4]})
    params.fix_param()
    assert params.input_names == ['input']

    # input names
    with pytest.raises(ValueError):
        params = BaseBackendParam(input_names=['input', 'input'])
        params.fix_param()

    with pytest.raises(ValueError):
        params = BaseBackendParam(input_names=['input', None])
        params.fix_param()

    # none name error
    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_names=['input', 'invalid'],
            input_shapes={None: [1, 2, 3, 4]})
        params.fix_param()

    # fill none name error
    with pytest.raises(ValueError):
        params = BaseBackendParam(input_shapes={None: [1, 2, 3, 4]})
        params.fix_param()

    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_names=['input'],
            input_shapes={
                None: [1, 2, 3, 4],
                'input': [1, 2, 3, 4]
            })
        params.fix_param()

    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_names=[None], input_shapes={None: [1, 2, 3, 4]})
        params.fix_param()

    # shape type error
    with pytest.raises(TypeError):
        params = BaseBackendParam(input_names=['input'], input_shapes=0)
        params.fix_param()

    with pytest.raises(TypeError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]}, min_shapes=0)
        params.fix_param()

    with pytest.raises(TypeError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]}, max_shapes=0)
        params.fix_param()

    # placeholder error
    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            min_shapes={'input': [1, 2, 3, None, None]})
        params.fix_param()

    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            max_shapes={'input': [1, 2, 3, None, None]})
        params.fix_param()


def test_check_param():

    params = BaseBackendParam(
        input_names=['input'], input_shapes={None: [1, 2, 3, 4]})
    params.check_param()

    #  input shapes != min/max shape
    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            min_shapes={'input': [1, 2, 3, 4, 5]})
        params.check_param()

    # different input names
    with pytest.raises(NameError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            min_shapes={'input1': [1, 2, 3, 4]})
        params.check_param()

    with pytest.raises(NameError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            max_shapes={'input1': [1, 2, 3, 4]})
        params.check_param()

    # shape type error
    with pytest.raises(TypeError):
        params = BaseBackendParam(input_shapes=0)
        params.check_param()

    with pytest.raises(TypeError):
        params = BaseBackendParam(min_shapes=0)
        params.check_param()

    with pytest.raises(TypeError):
        params = BaseBackendParam(max_shapes=0)
        params.check_param()

    # shape length error
    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            max_shapes={'input': [1, 2, 3, 4, 5]})
        params.check_param()

    # shape value error
    with pytest.raises(ValueError):
        params = BaseBackendParam(
            input_shapes={'input': [1, 2, 3, 4]},
            max_shapes={'input': [1, 1, 1, 1]})
        params.check_param()

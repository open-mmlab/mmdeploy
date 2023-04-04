# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdeploy.utils import docstring_parser as parser


@pytest.fixture(scope='module')
def singleline_arg():
    return '    input_names (list, optional): input_names. Defaults to None.'


@pytest.fixture(scope='module')
def multiline_arg():
    return (
        '    input_names (list, optional): input_names. Defaults to None.\n'
        '        next line.\n'
        '        next next line.\n')


def test_parse_arg(singleline_arg, multiline_arg):
    doc_arg, doc_len = parser.parse_arg(singleline_arg)
    assert doc_arg is not None
    assert doc_arg.name == 'input_names'
    assert doc_arg.type == 'list, optional'
    assert doc_arg.desc == 'input_names. Defaults to None.'
    assert doc_len == len(singleline_arg)

    doc_arg, doc_len = parser.parse_arg(multiline_arg)
    assert doc_arg is not None
    desc = 'input_names. Defaults to None.next line.next next line.'
    assert doc_arg.name == 'input_names'
    assert doc_arg.type == 'list, optional'
    assert doc_arg.desc == desc
    assert doc_len == len(multiline_arg)


@pytest.fixture(scope='module')
def doc_args():
    return (
        '    args (Any): args\n'
        '    input_names (list, optional): input_names. Defaults to None.\n'
        '            next line.\n'
        '            next next line.\n'
        '    output_names (list, optional): output_names. Defaults to None.\n'
        '    dynamic_axes (dict, optional): dynamic_axes. Defaults to None.\n'
        '    backend (str, optional): backend. Defaults to `onnxruntime`.\n'
        '\n\n'
        'Returns:\n'
        '    int: return val')


def test_parse_args(doc_args):
    doc_arg_list, doc_len = parser.parse_args(doc_args)
    assert len(doc_arg_list) == 5
    assert doc_len == len(doc_args) - len('\n\n'
                                          'Returns:\n'
                                          '    int: return val')

    gt_list = [
        ('args', 'Any', 'args'),
        ('input_names', 'list, optional',
         'input_names. Defaults to None.next line.next next line.'),
        ('output_names', 'list, optional', 'output_names. Defaults to None.'),
        ('dynamic_axes', 'dict, optional', 'dynamic_axes. Defaults to None.'),
        ('backend', 'str, optional', 'backend. Defaults to `onnxruntime`.'),
    ]

    for doc_arg, gt in zip(doc_arg_list, gt_list):
        gt_name, gt_type, gt_desc = gt
        assert doc_arg.name == gt_name
        assert doc_arg.type == gt_type
        assert doc_arg.desc == gt_desc


@pytest.fixture(scope='module')
def empty_lines():
    return ('    \n  \nnot empty')


def test_parse_empty_line(empty_lines):
    empty_len = parser.parse_empty_line(empty_lines)
    assert empty_len == len('    \n  \n')
    assert empty_lines[empty_len:] == 'not empty'


@pytest.fixture(scope='module')
def doc_args_section():
    return (
        'Args:   \n'
        '   \n    \n'
        '    args (Any): args\n'
        '    input_names (list, optional): input_names. Defaults to None.\n'
        '            next line.\n'
        '            next next line.\n'
        '    output_names (list, optional): output_names. Defaults to None.\n'
        '    dynamic_axes (dict, optional): dynamic_axes. Defaults to None.\n'
        '    backend (str, optional): backend. Defaults to `onnxruntime`.\n'
        '   \n \n')


def test_args_section(doc_args_section):
    doc_arg_list, doc_len = parser.parse_args_section(doc_args_section)
    assert len(doc_arg_list) == 5
    assert doc_len == len(doc_args_section)

    gt_list = [
        ('args', 'Any', 'args'),
        ('input_names', 'list, optional',
         'input_names. Defaults to None.next line.next next line.'),
        ('output_names', 'list, optional', 'output_names. Defaults to None.'),
        ('dynamic_axes', 'dict, optional', 'dynamic_axes. Defaults to None.'),
        ('backend', 'str, optional', 'backend. Defaults to `onnxruntime`.'),
    ]

    for doc_arg, gt in zip(doc_arg_list, gt_list):
        gt_name, gt_type, gt_desc = gt
        assert doc_arg.name == gt_name
        assert doc_arg.type == gt_type
        assert doc_arg.desc == gt_desc


@pytest.fixture(scope='module')
def full_doc_str():
    return (
        'This is Head\n'
        '\n'
        'This is desc1\n'
        'This is desc2\n'
        '\n'
        'Args:   \n'
        '   \n    \n'
        '    args (Any): args\n'
        '    input_names (list, optional): input_names. Defaults to None.\n'
        '            next line.\n'
        '            next next line.\n'
        '    output_names (list, optional): output_names. Defaults to None.\n'
        '    dynamic_axes (dict, optional): dynamic_axes. Defaults to None.\n'
        '    backend (str, optional): backend. Defaults to `onnxruntime`.\n'
        '   \n \n'
        'Returns:\n'
        '    int: return val')


def test_parse_docstring(full_doc_str):
    doc_str = parser.parse_docstring(full_doc_str)

    assert doc_str.head == 'This is Head'
    assert doc_str.desc == 'This is desc1\nThis is desc2'
    doc_arg_list = doc_str.args

    gt_list = [
        ('args', 'Any', 'args'),
        ('input_names', 'list, optional',
         'input_names. Defaults to None.next line.next next line.'),
        ('output_names', 'list, optional', 'output_names. Defaults to None.'),
        ('dynamic_axes', 'dict, optional', 'dynamic_axes. Defaults to None.'),
        ('backend', 'str, optional', 'backend. Defaults to `onnxruntime`.'),
    ]

    for doc_arg, gt in zip(doc_arg_list, gt_list):
        gt_name, gt_type, gt_desc = gt
        assert doc_arg.name == gt_name
        assert doc_arg.type == gt_type
        assert doc_arg.desc == gt_desc


class TestInspectDocstringArguments:

    @pytest.fixture(scope='class')
    def valid_obj(self):

        class ValidObj:
            """Valid Obj.

            Args:
                arg0 (int): description of arg0.
                arg1 (bool): description of arg1.
            """

        return ValidObj

    def test_inspect(self, valid_obj):
        assert parser.inspect_docstring_arguments(valid_obj) == [
            parser.DocStrArg(
                name='arg0', type='int', desc='description of arg0.'),
            parser.DocStrArg(
                name='arg1', type='bool', desc='description of arg1.')
        ]

        assert parser.inspect_docstring_arguments(
            valid_obj, ignore_args=['arg0']) == [
                parser.DocStrArg(
                    name='arg1', type='bool', desc='description of arg1.')
            ]

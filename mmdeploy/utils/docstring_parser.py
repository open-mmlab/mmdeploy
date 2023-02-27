# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class DocStrArg:
    """The argument of the docstring."""
    name: str
    type: str
    desc: str


@dataclass
class DocStr:
    """The packed docstring description."""
    head: str
    desc: str
    args: List[DocStrArg]


def parse_empty_line(buffer: str) -> int:
    """Get empty line in the buffer.

    Args:
        buffer (str): The string buffer.

    Returns:
        int: The length of the empty lines.
    """
    pattern = r'^[^\S\n]*(\n|$)'

    ret_len = 0
    while len(buffer) > 0:
        m = re.match(pattern, buffer)
        if m is None:
            break
        next_len = m.end() - m.start()
        ret_len += next_len
        buffer = buffer[next_len:]

    return ret_len


def parse_arg(buffer: str) -> Tuple[Optional[DocStrArg], int]:
    """Parse one argument at the beginning of the buffer.

    Args:
        buffer (str): The docstring buffer

    Returns:
        Tuple[Optional[DocStrArg], int]: The argument info and the
            buffer length. If there is no argument, return (None, 0)
    """
    ret_len = 0
    pattern = r'^[^\S\n]*(?P<name>\w+)[^\S\n]*\((?P<type>.*)\):[^\S\n]*(?P<desc>.*)(\n|$)'  # noqa
    m = re.match(pattern, buffer)
    if m is None:
        return None, 0

    doc_dict = m.groupdict()
    next_len = m.end() - m.start()
    buffer = buffer[next_len:]
    ret_len += next_len

    # try read docs with multiline
    while len(buffer) > 0:
        # if next line is starts with same pattern, it is not remain
        m = re.match(pattern, buffer)
        if m is not None:
            break

        # remain should starts with whitespace.
        m = re.match(r'^[^\S\n]+(?P<desc>\S.*)(\n|$)', buffer)
        if m is None:
            break
        remain_desc = m.group('desc')
        doc_dict['desc'] += remain_desc
        next_len = m.end() - m.start()
        buffer = buffer[next_len:]
        ret_len += next_len

    doc_arg = DocStrArg(**doc_dict)
    return doc_arg, ret_len


def parse_args(buffer: str) -> Tuple[List[DocStrArg], int]:
    """Parse all arguments at the beginning of the buffer.

    Args:
        buffer (str): The docstring buffer

    Returns:
        Tuple[List[DocStrArg], int]: The list of all parsed arguments and
            buffer length.
    """
    ret_args = []
    ret_len = 0
    while len(buffer) > 0:
        doc_arg, doc_len = parse_arg(buffer)
        if doc_len == 0 or doc_arg is None:
            break
        ret_args.append(doc_arg)
        buffer = buffer[doc_len:]
        ret_len += doc_len

    return ret_args, ret_len


def parse_args_section(buffer: str) -> Tuple[List[DocStrArg], int]:
    """Parse The arguments section, with the head `Args:`

    Args:
        buffer (str): The docstring buffer

    Returns:
        Tuple[List[DocStrArg], int]: The list of all parsed arguments and
            buffer length.
    """
    ret_len = 0

    def _update_buffer(next_len: int):
        nonlocal buffer
        nonlocal ret_len
        buffer = buffer[next_len:]
        ret_len += next_len

    def _skip_empty_line():
        nonlocal buffer
        next_len = parse_empty_line(buffer)
        _update_buffer(next_len)

    # parse `Args:`
    head_pattern = r'^Args:\s*(\n|$)'
    m = re.match(head_pattern, buffer)
    if m is None:
        return [], 0
    next_len = m.end() - m.start()
    _update_buffer(next_len)
    _skip_empty_line()

    # parse args
    doc_args, next_len = parse_args(buffer)
    _update_buffer(next_len)
    _skip_empty_line()

    return doc_args, ret_len


SECTION_HEAD = ['Args:', 'Returns:', 'Examples:']


def parse_desc(buffer: str) -> Tuple[str, int]:
    """Parse the description of the docstring.

    Args:
        buffer (str): The docstring buffer

    Returns:
        Tuple[str, int]: The description string and the buffer length
    """
    desc_pattern = r'^(?P<desc>.*(\n|$))'
    desc = ''
    desc_len = 0
    while len(buffer) > 0:
        m = re.match(desc_pattern, buffer)
        if m is None:
            break
        line_desc = m.group('desc')

        # check if buffer reach next section
        if line_desc.rstrip() in SECTION_HEAD:
            break

        line_len = m.end() - m.start()
        desc += line_desc
        desc_len += line_len
        buffer = buffer[line_len:]

    return desc.strip(), desc_len


def parse_docstring(buffer: str) -> Optional[DocStr]:
    """Parse docstring.

    Args:
        buffer (str): The docstring buffer

    Returns:
        Optional[DocStr]: The parsed docstring info. If parse failed. return
            None.
    """
    head = ''
    desc = ''
    args = []

    def _skip_empty_line():
        nonlocal buffer
        next_len = parse_empty_line(buffer)
        buffer = buffer[next_len:]

    # parse head
    head_pattern = r'^(?P<head>.*)(\n|$)'
    m = re.match(head_pattern, buffer)
    if m is None:
        return DocStr(head=head, desc=desc, args=args)
    next_len = m.end() - m.start()
    head = m.group('head').rstrip()
    buffer = buffer[next_len:]
    _skip_empty_line()

    # parse desc
    desc, next_len = parse_desc(buffer)
    buffer = buffer[next_len:]

    while len(buffer) > 0:
        section_pattern = r'^(?P<section>.*)(\n|$)'
        m = re.match(section_pattern, buffer)
        if m is None:
            return DocStr(head=head, desc=desc, args=args)

        if m.group('section').rstrip() == 'Args:':
            args, next_len = parse_args_section(buffer)
        else:
            next_len = m.end() - m.start()

        buffer = buffer[next_len:]

    return DocStr(head=head, desc=desc, args=args)


def inspect_docstring_arguments(obj: Any,
                                ignore_args: Optional[List[str]] = None
                                ) -> List[DocStrArg]:
    """Inspect google style function docstring.

    Args:
        obj (Any): Object with docstring
        ignore_args (List[str]): Ignore arguments when parsing.

    Returns:
        List[DocStrArg]: Parsed docstring arguments.
    """
    ignore_args = [] if ignore_args is None else ignore_args
    doc_str: DocStr = parse_docstring(inspect.getdoc(obj))

    def _filter_cb(arg: DocStrArg):
        nonlocal ignore_args
        return arg.name not in ignore_args

    return list(filter(_filter_cb, doc_str.args))

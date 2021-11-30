# Copyright (c) OpenMMLab. All rights reserved.
from .extractor import create_extractor, parse_extractor_io_string
from .function_marker import mark, reset_mark_function_count
from .optimize import (attribute_to_dict, get_new_name, remove_identity,
                       rename_value)

__all__ = [
    'mark', 'reset_mark_function_count', 'create_extractor',
    'parse_extractor_io_string', 'remove_identity', 'attribute_to_dict',
    'rename_value', 'get_new_name'
]

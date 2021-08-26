from .getattribute import getattribute_static
from .interpolate import interpolate_static
from .repeat import repeat_static
from .size import size_of_tensor_static
from .topk import topk_dynamic, topk_static

__all__ = [
    'getattribute_static', 'interpolate_static', 'repeat_static',
    'size_of_tensor_static', 'topk_static', 'topk_dynamic'
]

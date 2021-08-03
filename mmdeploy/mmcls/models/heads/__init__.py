from .cls_head import simple_test_of_cls_head
from .linear_head import simple_test_of_linear_head
from .multi_label_head import simple_test_of_multi_label_head
from .multi_label_linear_head import simple_test_of_multi_label_linear_head
from .stacked_head import simple_test_of_stacked_head
from .vision_transformer_head import simple_test_of_vision_transformer_head

__all__ = [
    'simple_test_of_multi_label_linear_head',
    'simple_test_of_multi_label_head', 'simple_test_of_cls_head',
    'simple_test_of_linear_head', 'simple_test_of_stacked_head',
    'simple_test_of_vision_transformer_head'
]

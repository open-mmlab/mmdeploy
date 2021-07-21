from .anchor_head import get_bboxes_of_anchor_head
from .fcos_head import get_bboxes_of_fcos_head
from .rpn_head import get_bboxes_of_rpn_head

__all__ = [
    'get_bboxes_of_anchor_head', 'get_bboxes_of_fcos_head',
    'get_bboxes_of_rpn_head'
]

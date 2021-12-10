# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.rpn.RPN.simple_test')
def rpn__simple_test(ctx, self, img, img_metas, **kwargs):
    """Rewrite `simple_test` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (list[dict]): Dict containing image's meta information
            such as `img_shape`.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
    """
    x = self.extract_feat(img)
    return self.rpn_head.simple_test_rpn(x, img_metas)

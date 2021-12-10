# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.two_stage.TwoStageDetector.extract_feat')
@mark('extract_feat', inputs='img', outputs='feat')
def two_stage_detector__extract_feat(ctx, self, img):
    """Rewrite `extract_feat` for default backend.

    This function uses the specific `extract_feat` function for the two
    stage detector after adding marks.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).

    Returns:
        list[Tensor]: Each item with shape (N, C, H, W) corresponds one
        level of backbone and neck features.
    """
    return ctx.origin_func(self, img)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.two_stage.TwoStageDetector.simple_test')
def two_stage_detector__simple_test(ctx,
                                    self,
                                    img,
                                    img_metas,
                                    proposals=None,
                                    **kwargs):
    """Rewrite `simple_test` for default backend.

    Support configured dynamic/static shape for model input and return
    detection result as Tensor instead of numpy array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (list[dict]): Dict containing image's meta information
            such as `img_shape`.
        proposals (List[Tensor]): Region proposals.
            Default is None.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
    """
    assert self.with_bbox, 'Bbox head must be implemented.'
    x = self.extract_feat(img)
    if proposals is None:
        proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
    return self.roi_head.simple_test(x, proposals, img_metas, rescale=False)

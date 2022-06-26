# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.panoptic_two_stage_segmentor.'
    'TwoStagePanopticSegmentor.simple_test')
def two_stage_panoptic_segmentor__simple_test(ctx,
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
        tuple[Tensor, Tensor, Tensor, Tensor]:
            (bboxes, labels, masks, semseg), `bboxes` of shape [N, num_det, 5],
            `labels` of shape [N, num_det], `masks` of shape [N, roi_H, roi_W],
            `semseg` of shape [N, num_sem_class, sem_H, sem_W].
    """
    assert self.with_bbox, 'Bbox head must be implemented.'
    x = self.extract_feat(img)
    if proposals is None:
        proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)

    bboxes, labels, masks = self.roi_head.simple_test(
        x, proposals, img_metas, rescale=False)

    semseg = self.semantic_head.simple_test(x, img_metas, rescale=False)
    return bboxes, labels, masks, semseg

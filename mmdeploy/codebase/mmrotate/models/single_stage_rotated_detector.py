# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.models.detectors.RotatedSingleStageDetector'
    '.simple_test')
def single_stage_rotated_detector__simple_test(ctx,
                                               self,
                                               img,
                                               img_metas,
                                               rescale=False):
    """Rewrite `simple_test` of RotatedSingleStageDetector for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class
            SingleStageTextDetector.
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.

    Returns:
        outs (Tensor): A feature map output from bbox_head. The tensor shape
            (N, C, H, W).
    """
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    outs = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

    return outs

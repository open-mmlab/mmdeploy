import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmdet.core import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.BBoxHead.get_bboxes')
def get_bboxes_of_bbox_head(ctx, self, rois, cls_score, bbox_pred, img_shape,
                            cfg, **kwargs):

    assert rois.ndim == 3, 'Only support export two stage ' \
                           'model to ONNX ' \
                           'with batch dimension. '
    if self.custom_cls_channels:
        scores = self.loss_cls.get_activation(cls_score)
    else:
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None

    if bbox_pred is not None:
        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)
    else:
        bboxes = rois[..., 1:].clone()
        if img_shape is not None:
            max_shape = bboxes.new_tensor(img_shape)[..., :2]
            min_xy = bboxes.new_tensor(0)
            max_xy = torch.cat([max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
            bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
            bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    batch_size = scores.shape[0]
    device = scores.device
    # ignore background class
    scores = scores[..., :self.num_classes]
    if not self.reg_class_agnostic:
        # only keep boxes with the max scores
        max_inds = scores.reshape(-1, self.num_classes).argmax(1, keepdim=True)
        bboxes = bboxes.reshape(-1, self.num_classes, 4)
        dim0_inds = torch.arange(
            bboxes.shape[0], device=device).view(-1, 1).expand_as(max_inds)
        bboxes = bboxes[dim0_inds, max_inds].reshape(batch_size, -1, 4)

    # get nms params
    post_params = ctx.cfg.post_processing
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    dets, labels = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)

    return dets, labels

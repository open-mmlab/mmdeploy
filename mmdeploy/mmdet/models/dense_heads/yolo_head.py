import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmdet.core import multiclass_nms
from mmdeploy.mmdet.export import pad_with_value
from mmdeploy.utils import (Backend, get_backend, get_mmdet_params,
                            is_dynamic_shape)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.YOLOV3Head.get_bboxes')
def yolov3_head__get_bboxes(ctx,
                            self,
                            pred_maps,
                            with_nms=True,
                            cfg=None,
                            **kwargs):
    """Rewrite `get_bboxes` for default backend.

    Transform network output for a batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        self: Represent the instance of the original class.
        pred_maps (list[Tensor]): Raw predictions for a batch of images.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    is_dynamic_flag = is_dynamic_shape(ctx.cfg)
    num_levels = len(pred_maps)
    pred_maps_list = [pred_maps[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(pred_maps_list) == self.num_levels

    device = pred_maps_list[0].device
    batch_size = pred_maps_list[0].shape[0]

    featmap_sizes = [
        pred_maps_list[i].shape[-2:] for i in range(self.num_levels)
    ]
    multi_lvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device)
    pre_topk = cfg.get('nms_pre', -1)
    multi_lvl_bboxes = []
    multi_lvl_cls_scores = []
    multi_lvl_conf_scores = []
    for i in range(self.num_levels):
        # get some key info for current scale
        pred_map = pred_maps_list[i]
        stride = self.featmap_strides[i]
        # (b,h, w, num_anchors*num_attrib) ->
        # (b,h*w*num_anchors, num_attrib)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(batch_size, -1, self.num_attrib)
        # Inplace operation like
        # ```pred_map[..., :2] = \torch.sigmoid(pred_map[..., :2])```
        # would create constant tensor when exporting to onnx
        pred_map_conf = torch.sigmoid(pred_map[..., :2])
        pred_map_rest = pred_map[..., 2:]
        pred_map = torch.cat([pred_map_conf, pred_map_rest], dim=-1)
        pred_map_boxes = pred_map[..., :4]
        multi_lvl_anchor = multi_lvl_anchors[i]
        # use static anchor if input shape is static
        if not is_dynamic_flag:
            multi_lvl_anchor = multi_lvl_anchor.data
        multi_lvl_anchor = multi_lvl_anchor.unsqueeze(0).expand_as(
            pred_map_boxes)
        bbox_pred = self.bbox_coder.decode(multi_lvl_anchor, pred_map_boxes,
                                           stride)
        # conf and cls
        conf_pred = torch.sigmoid(pred_map[..., 4])
        cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
            batch_size, -1, self.num_classes)  # Cls pred one-hot.

        backend = get_backend(ctx.cfg)
        # topk in tensorrt does not support shape<k
        # concate zero to enable topk,
        if backend == Backend.TENSORRT:
            bbox_pred = pad_with_value(bbox_pred, 1, pre_topk)
            conf_pred = pad_with_value(conf_pred, 1, pre_topk, 0.)
            cls_pred = pad_with_value(cls_pred, 1, pre_topk, 0.)

        if pre_topk > 0:
            _, topk_inds = conf_pred.topk(pre_topk)
            batch_inds = torch.arange(
                batch_size, device=device).view(-1,
                                                1).expand_as(topk_inds).long()
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
            transformed_inds = (bbox_pred.shape[1] * batch_inds + topk_inds)
            bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds, :].reshape(
                batch_size, -1, 4)
            cls_pred = cls_pred.reshape(
                -1, self.num_classes)[transformed_inds, :].reshape(
                    batch_size, -1, self.num_classes)
            conf_pred = conf_pred.reshape(-1, 1)[transformed_inds].reshape(
                batch_size, -1)

        # Save the result of current scale
        multi_lvl_bboxes.append(bbox_pred)
        multi_lvl_cls_scores.append(cls_pred)
        multi_lvl_conf_scores.append(conf_pred)

    # Merge the results of different scales together
    batch_mlvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(multi_lvl_cls_scores, dim=1)
    batch_mlvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)

    post_params = get_mmdet_params(ctx.cfg)

    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    confidence_threshold = cfg.get('conf_thr',
                                   post_params.confidence_threshold)

    # follow original pipeline of YOLOv3
    if confidence_threshold > 0:
        mask = (batch_mlvl_conf_scores >= confidence_threshold).float()
        batch_mlvl_conf_scores *= mask
    if score_threshold > 0:
        mask = (batch_mlvl_scores > score_threshold).float()
        batch_mlvl_scores *= mask

    batch_mlvl_conf_scores = batch_mlvl_conf_scores.unsqueeze(2).expand_as(
        batch_mlvl_scores)
    batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_conf_scores

    if with_nms:
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        # keep aligned with original pipeline, improve
        # mAP by 1% for YOLOv3 in ONNX
        score_threshold = 0
        return multiclass_nms(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        return batch_mlvl_bboxes, batch_mlvl_scores


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.YOLOV3Head.get_bboxes', backend='ncnn')
def yolov3_head__get_bboxes__ncnn(ctx,
                                  self,
                                  pred_maps,
                                  with_nms=True,
                                  cfg=None,
                                  **kwargs):
    """Rewrite `get_bboxes` for ncnn backend.

    Transform network output for a batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        self: Represent the instance of the original class.
        pred_maps (list[Tensor]): Raw predictions for a batch of images.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    num_levels = len(pred_maps)
    pred_maps_list = [pred_maps[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(pred_maps_list) == self.num_levels

    device = pred_maps_list[0].device
    batch_size = pred_maps_list[0].shape[0]

    featmap_sizes = [
        pred_maps_list[i].shape[-2:] for i in range(self.num_levels)
    ]
    multi_lvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device)
    pre_topk = cfg.get('nms_pre', -1)
    multi_lvl_bboxes = []
    multi_lvl_cls_scores = []
    multi_lvl_conf_scores = []
    for i in range(self.num_levels):
        # get some key info for current scale
        pred_map = pred_maps_list[i]
        stride = self.featmap_strides[i]
        # (b,h, w, num_anchors*num_attrib) ->
        # (b,h*w*num_anchors, num_attrib)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(batch_size, -1, self.num_attrib)
        # Inplace operation like
        # ```pred_map[..., :2] = \torch.sigmoid(pred_map[..., :2])```
        # would create constant tensor when exporting to onnx
        pred_map_conf = torch.sigmoid(pred_map[..., :2])
        pred_map_rest = pred_map[..., 2:]
        # dim must be written as 2, but not -1, because ncnn implicit batch
        # mechanism.
        pred_map = torch.cat([pred_map_conf, pred_map_rest], dim=2)
        pred_map_boxes = pred_map[..., :4]
        multi_lvl_anchor = multi_lvl_anchors[i]
        # use static anchor if input shape is static
        multi_lvl_anchor = multi_lvl_anchor.unsqueeze(0).expand_as(
            pred_map_boxes).data

        bbox_pred = self.bbox_coder.decode(multi_lvl_anchor, pred_map_boxes,
                                           stride)
        # conf and cls
        conf_pred = torch.sigmoid(pred_map[..., 4])
        cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
            batch_size, -1, self.num_classes)  # Cls pred one-hot.

        if pre_topk > 0:
            _, topk_inds = conf_pred.topk(pre_topk)
            topk_inds = topk_inds.view(-1)
            bbox_pred = bbox_pred[:, topk_inds, :]
            cls_pred = cls_pred[:, topk_inds, :]
            conf_pred = conf_pred[:, topk_inds]

        # Save the result of current scale
        multi_lvl_bboxes.append(bbox_pred)
        multi_lvl_cls_scores.append(cls_pred)
        multi_lvl_conf_scores.append(conf_pred)

    # Merge the results of different scales together
    batch_mlvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(multi_lvl_cls_scores, dim=1)
    batch_mlvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)

    post_params = get_mmdet_params(ctx.cfg)

    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    confidence_threshold = cfg.get('conf_thr',
                                   post_params.confidence_threshold)

    # helper function for creating Threshold op
    def _create_threshold(x, thresh):

        class ThresholdOp(torch.autograd.Function):
            """Create Threshold op."""

            @staticmethod
            def forward(ctx, x, threshold):
                return x > threshold

            @staticmethod
            def symbolic(g, x, threshold):
                return g.op(
                    'mmdeploy::Threshold', x, threshold_f=threshold, outputs=1)

        return ThresholdOp.apply(x, thresh)

    # follow original pipeline of YOLOv3
    if confidence_threshold > 0:
        mask = _create_threshold(batch_mlvl_conf_scores,
                                 confidence_threshold).float()
        batch_mlvl_conf_scores *= mask
    if score_threshold > 0:
        mask = _create_threshold(batch_mlvl_scores, score_threshold).float()
        batch_mlvl_scores *= mask

    # NCNN broadcast needs the same in channel dimension.
    _batch_mlvl_conf_scores = batch_mlvl_conf_scores.unsqueeze(2).unsqueeze(3)
    _batch_mlvl_scores = batch_mlvl_scores.unsqueeze(3)
    batch_mlvl_scores = (_batch_mlvl_scores * _batch_mlvl_conf_scores).reshape(
        batch_mlvl_scores.shape)
    # Although batch_mlvl_bboxes already has the shape of
    # (batch_size, -1, 4), ncnn implicit batch mechanism in the model and
    # ncnn channel alignment would result in a shape of
    # (batch_size, -1, 4, 1). So, we need a reshape op to ensure the
    # batch_mlvl_bboxes shape is right.
    batch_mlvl_bboxes = batch_mlvl_bboxes.reshape(batch_size, -1, 4)

    if with_nms:
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        # keep aligned with original pipeline, improve
        # mAP by 1% for YOLOv3 in ONNX
        score_threshold = 0
        return multiclass_nms(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        return batch_mlvl_bboxes, batch_mlvl_scores

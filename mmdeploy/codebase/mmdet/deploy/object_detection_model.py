# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData
from torch import Tensor, nn

from mmdeploy.backend.base import get_backend_file_count
from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_partition_config, load_config)

# Use registry to store models with different partition methods
# If a model doesn't need to partition, we don't need this registry
__BACKEND_MODEL = Registry('backend_detectors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        class_names (Sequence[str]): A list of string specifying class names.
        deploy_cfg (str|Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    @staticmethod
    def __clear_outputs(
        test_outputs: List[Union[Tensor, np.ndarray]]
    ) -> List[Union[List[Tensor], List[np.ndarray]]]:
        """Removes additional outputs and detections with zero and negative
        score.

        Args:
            test_outputs (List[Union[Tensor, np.ndarray]]):
                outputs of forward_test.

        Returns:
            List[Union[List[Tensor], List[np.ndarray]]]:
                outputs with without zero score object.
        """
        batch_size = len(test_outputs[0])

        num_outputs = len(test_outputs)
        outputs = [[None for _ in range(batch_size)]
                   for _ in range(num_outputs)]

        for i in range(batch_size):
            inds = test_outputs[0][i, :, 4] > 0.0
            for output_id in range(num_outputs):
                outputs[output_id][i] = test_outputs[output_id][i, inds, ...]
        return outputs

    @staticmethod
    def postprocessing_masks(det_bboxes: Union[np.ndarray, Tensor],
                             det_masks: Union[np.ndarray, Tensor],
                             img_w: int,
                             img_h: int,
                             device: str = 'cpu') -> Tensor:
        """Additional processing of masks. Resizes masks from [num_det, 28, 28]
        to [num_det, img_w, img_h]. Analog of the 'mmdeploy.codebase.mmdet.
        models.roi_heads.fcn_mask_head._do_paste_mask' function.

        Args:
            det_bboxes (np.ndarray | Tensor): Bbox of shape [num_det, 4]
            det_masks (np.ndarray | Tensor): Masks of shape [num_det, 28, 28].
            img_w (int): Width of the original image.
            img_h (int): Height of the original image.
            device :(str): The device type.

        Returns:
            Tensor: masks of shape [N, num_det, img_h, img_w].
        """
        masks = det_masks
        bboxes = det_bboxes
        device = torch.device(device)
        num_det = bboxes.shape[0]
        # Skip postprocessing if no detections are found.
        if num_det == 0:
            return torch.zeros(
                0, img_h, img_w, dtype=torch.float32, device=device)

        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks, device=device)
            bboxes = torch.tensor(bboxes, device=device)

        masks = masks.to(device)
        bboxes = bboxes.to(device)

        result_masks = []
        for bbox, mask in zip(bboxes, masks):

            x0_int, y0_int = 0, 0
            x1_int, y1_int = img_w, img_h

            img_y = torch.arange(
                y0_int, y1_int, dtype=torch.float32, device=device) + 0.5
            img_x = torch.arange(
                x0_int, x1_int, dtype=torch.float32, device=device) + 0.5
            x0, y0, x1, y1 = bbox

            img_y = (img_y - y0) / (y1 - y0) * 2 - 1
            img_x = (img_x - x0) / (x1 - x0) * 2 - 1
            if torch.isinf(img_x).any():
                inds = torch.where(torch.isinf(img_x))
                img_x[inds] = 0
            if torch.isinf(img_y).any():
                inds = torch.where(torch.isinf(img_y))
                img_y[inds] = 0

            gx = img_x[None, :].expand(img_y.size(0), img_x.size(0))
            gy = img_y[:, None].expand(img_y.size(0), img_x.size(0))
            grid = torch.stack([gx, gy], dim=2)

            img_masks = F.grid_sample(
                mask.to(dtype=torch.float32)[None, None, :, :],
                grid[None, :, :, :],
                align_corners=False)

            result_masks.append(img_masks)
        result_masks = torch.cat(result_masks, 1)
        return result_masks.squeeze(0)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                **kwargs) -> Any:
        assert mode == 'predict', 'Deploy model only allow mode=="predict".'
        inputs = inputs.contiguous()
        outputs = self.predict(inputs)
        outputs = End2EndModel.__clear_outputs(outputs)
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = inputs.shape[0]
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            result = InstanceData()

            bboxes = dets[:, :4]
            scores = dets[:, 4]

            # perform rescale
            if rescale:
                scale_factor = img_metas[i]['scale_factor']
                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    if len(scale_factor) == 2:
                        scale_factor = np.array(scale_factor)
                        scale_factor = np.concatenate(
                            [scale_factor, scale_factor])
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                scale_factor = torch.from_numpy(scale_factor).to(dets)
                bboxes /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                bboxes[:, ::2] -= x_off
                bboxes[:, 1::2] -= y_off
                bboxes *= (bboxes > 0)

            result.scores = scores
            result.bboxes = bboxes
            result.labels = labels

            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                export_postprocess_mask = True
                if self.deploy_cfg is not None:

                    mmdet_deploy_cfg = get_post_processing_params(
                        self.deploy_cfg)
                    # this flag enable postprocess when export.
                    export_postprocess_mask = mmdet_deploy_cfg.get(
                        'export_postprocess_mask', True)
                if not export_postprocess_mask:
                    masks = End2EndModel.postprocessing_masks(
                        dets[:, :4], masks, ori_w, ori_h, self.device)
                else:
                    masks = masks[:, :img_h, :img_w]
                # avoid to resize masks with zero dim
                if rescale and masks.shape[0] != 0:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0), size=(ori_h, ori_w))
                    masks = masks.squeeze(0)
                if masks.dtype != bool:
                    masks = masks >= 0.5
                # aligned with mmdet to easily convert to numpy
                masks = masks.cpu()
                result.masks = masks
            data_samples[i].pred_instances = result
            results.append(data_samples[i])
        return results

    def predict(self, imgs: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs


@__BACKEND_MODEL.register_module('single_stage')
class PartitionSingleStageModel(End2EndModel):
    """Partitioned single stage detection model.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg (str|Config): Input model config file or Config
            object.
        deploy_cfg (str|Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self, backend: Backend, backend_files: Sequence[str],
                 device: str, class_names: Sequence[str],
                 model_cfg: Union[str, Config], deploy_cfg: Union[str, Config],
                 **kwargs):
        super().__init__(backend, backend_files, device, class_names,
                         deploy_cfg, **kwargs)
        # load cfg if necessary
        model_cfg = load_config(model_cfg)[0]
        self.model_cfg = model_cfg

    def _init_wrapper(self, backend, backend_files, device):
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=['scores', 'boxes'],
            deploy_cfg=self.deploy_cfg)

    def partition0_postprocess(self, scores: Tensor, bboxes: Tensor):
        """Perform post-processing for partition 0.

        Args:
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            bboxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        cfg = self.model_cfg.model.test_cfg
        deploy_cfg = self.deploy_cfg

        post_params = get_post_processing_params(deploy_cfg)
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = -1 if post_params.pre_top_k >= bboxes.shape[1] \
            else post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        ret = multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
        ret = [r.cpu() for r in ret]
        return ret

    def forward_test(self, imgs: Tensor, *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            list[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        scores, bboxes = outputs[:2]
        return self.partition0_postprocess(scores, bboxes)


@__BACKEND_MODEL.register_module('two_stage')
class PartitionTwoStageModel(End2EndModel):
    """Partitioned two stage detection model.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg (str|Config): Input model config file or Config
            object.
        deploy_cfg (str|Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self, backend: Backend, backend_files: Sequence[str],
                 device: str, class_names: Sequence[str],
                 model_cfg: Union[str, Config], deploy_cfg: Union[str, Config],
                 **kwargs):

        # load cfg if necessary
        model_cfg = load_config(model_cfg)[0]

        self.model_cfg = model_cfg

        super().__init__(backend, backend_files, device, class_names,
                         deploy_cfg, **kwargs)
        from mmdet.models.builder import build_head, build_roi_extractor

        from ..models.roi_heads.bbox_head import bbox_head__get_bboxes

        self.bbox_roi_extractor = build_roi_extractor(
            model_cfg.model.roi_head.bbox_roi_extractor)
        self.bbox_head = build_head(model_cfg.model.roi_head.bbox_head)

        class Context:
            pass

        ctx = Context()
        ctx.cfg = self.deploy_cfg
        self.bbox_head__get_bboxes = partial(bbox_head__get_bboxes, ctx)

    def _init_wrapper(self, backend, backend_files, device):
        n = get_backend_file_count(backend)
        num_feat = self.model_cfg['model']['neck']['num_outs']
        partition0_output_names = [
            'feat/{}'.format(i) for i in range(num_feat)
        ] + ['scores', 'boxes']

        self.first_wrapper = BaseBackendModel._build_wrapper(
            backend,
            backend_files[0:n],
            device,
            output_names=partition0_output_names,
            deploy_cfg=self.deploy_cfg)

        self.second_wrapper = BaseBackendModel._build_wrapper(
            backend,
            backend_files[n:2 * n],
            device,
            output_names=['cls_score', 'bbox_pred'],
            deploy_cfg=self.deploy_cfg)

    def partition0_postprocess(self, x: Sequence[Tensor], scores: Tensor,
                               bboxes: Tensor):
        """Perform post-processing for partition 0.

        Args:
            x (tuple[Tensor]): Feature maps of all scale levels.
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            bboxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].

        Returns:
            tuple(Tensor, Tensor): rois and bbox_feats.
        """
        # rpn-nms + roi-extractor
        cfg = self.model_cfg.model.test_cfg.rpn
        deploy_cfg = self.deploy_cfg

        post_params = get_post_processing_params(deploy_cfg)
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = -1 if post_params.pre_top_k >= bboxes.shape[1] \
            else post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        # only one class in rpn
        max_output_boxes_per_class = keep_top_k
        proposals, _ = multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)

        rois = proposals
        batch_index = torch.arange(
            rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        return rois, bbox_feats

    def partition1_postprocess(self, rois: Tensor, cls_score: Tensor,
                               bbox_pred: Tensor, img_metas: Sequence[dict]):
        """Perform post-processing for partition 1.
        Args:
            rois (Tensor): Input tensor of roi.
            cls_score (Tensor): Scores of all classes.
            bbox_pred (Tensor): Bounding box proposals.
            img_metas (Sequence[dict]): A list of image(s) meta information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class
                labels of shape [N, num_det].
        """
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))

        rcnn_test_cfg = self.model_cfg.model.test_cfg.rcnn
        return self.bbox_head__get_bboxes(
            self.bbox_head,
            rois,
            cls_score,
            bbox_pred,
            img_metas[0][0]['img_shape'],
            img_metas[0][0]['scale_factor'],
            cfg=rcnn_test_cfg)

    def forward_test(self, imgs: Tensor, img_metas: Sequence[dict], *args,
                     **kwargs):
        """Implement forward test.

        Args:
            imgs (Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of image(s) meta information.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        outputs = self.first_wrapper({'input': imgs})
        outputs = self.first_wrapper.output_to_list(outputs)
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # partition0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # partition1 forward
        bbox_feats = bbox_feats.contiguous()
        outputs = self.second_wrapper({'bbox_feats': bbox_feats})
        outputs = self.second_wrapper.output_to_list(outputs)
        cls_score, bbox_pred = outputs[:2]

        # partition1_postprocess
        outputs = self.partition1_postprocess(rois, cls_score, bbox_pred,
                                              img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


@__BACKEND_MODEL.register_module('ncnn_end2end')
class NCNNEnd2EndModel(End2EndModel):
    """NCNNEnd2EndModel.

    End2end NCNN model inference class. Because it has DetectionOutput layer
    and its output is different from original mmdet style of `dets`, `labels`.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg (str|Config): Input model config file or Config
            object.
        deploy_cfg (str|Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self, backend: Backend, backend_files: Sequence[str],
                 device: str, model_cfg: Union[str, Config],
                 deploy_cfg: Union[str, Config], **kwargs):
        assert backend == Backend.NCNN, f'only supported ncnn, but give \
            {backend.value}'

        super(NCNNEnd2EndModel, self).__init__(backend, backend_files, device,
                                               deploy_cfg, **kwargs)
        # load cfg if necessary
        model_cfg = load_config(model_cfg)[0]
        self.model_cfg = model_cfg

    def predict(self, imgs: Tensor, *args, **kwargs) -> List:
        """Implement forward test.

        Args:
            imgs (Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            list[Tensor]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        _, _, H, W = imgs.shape
        outputs = self.wrapper({self.input_name: imgs})
        for key, item in outputs.items():
            if item is None:
                return torch.zeros(1, 0, 5), torch.zeros(1, 0)
        out = self.wrapper.output_to_list(outputs)[0]
        labels = out[:, :, 0] - 1
        scales = torch.tensor([W, H, W, H]).reshape(1, 1, 4).to(out)
        scores = out[:, :, 1:2]
        boxes = out[:, :, 2:6] * scales
        dets = torch.cat([boxes, scores], dim=2)
        return dets, torch.tensor(labels, dtype=torch.int32)


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmdet format."""

    def __init__(self, *args, **kwargs):
        kwargs['data_preprocessor'] = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)
        self.has_mask = self.deploy_cfg.codebase_config.get('has_mask', False)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict'):
        """Run forward inference.

        Args:
            img (Sequence[Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of meta info for image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        if isinstance(inputs, list):
            inputs = inputs[0]
        # inputs are c,h,w, sdk requested h,w,c
        inputs = inputs.permute(1, 2, 0)
        dets, labels, masks = self.wrapper.invoke(
            inputs.contiguous().detach().cpu().numpy())
        result = InstanceData()
        if self.has_mask:
            segm_results = []
            ori_h, ori_w = data_samples[0].ori_shape[:2]
            for bbox, mask in zip(dets, masks):
                img_mask = np.zeros((ori_h, ori_w), dtype=np.uint8)
                left = int(max(np.floor(bbox[0]) - 1, 0))
                top = int(max(np.floor(bbox[1]) - 1, 0))
                img_mask[top:top + mask.shape[0],
                         left:left + mask.shape[1]] = mask
                segm_results.append(torch.from_numpy(img_mask))
            if len(segm_results) > 0:
                result.masks = torch.stack(segm_results, 0).to(self.device)
            else:
                result.masks = torch.zeros([0, ori_h, ori_w]).to(self.device)
        dets = torch.from_numpy(dets).to(self.device)
        labels = torch.from_numpy(labels).to(torch.int64).to(self.device)
        result.bboxes = dets[:, :4]
        result.scores = dets[:, 4]
        result.labels = labels
        data_samples[0].pred_instances = result

        return data_samples


def build_object_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        End2EndModel: Detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)

    partition_config = get_partition_config(deploy_cfg)
    if partition_config is not None:
        partition_type = partition_config.get('type', None)
    else:
        codebase_config = get_codebase_config(deploy_cfg)
        # Default Config is 'end2end'
        partition_type = codebase_config.get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=partition_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector

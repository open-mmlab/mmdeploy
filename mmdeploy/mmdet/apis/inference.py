from functools import partial
from typing import List, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result
from mmdet.datasets import DATASETS
from mmdet.models import BaseDetector

from mmdeploy.mmdet.core.post_processing import multiclass_nms
from mmdeploy.utils import (Backend, get_backend, get_mmdet_params,
                            get_partition_config, load_config)


class DeployBaseDetector(BaseDetector):
    """Base Class of Wrapper for inference of detection.

    Args:
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, class_names, device_id, deploy_cfg=None, **kwargs):
        super(DeployBaseDetector, self).__init__()
        self.CLASSES = class_names
        self.device_id = device_id
        self.deploy_cfg = deploy_cfg

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def val_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def train_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def aforward_test(self, *, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def __clear_outputs(
        self, test_outputs: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[Union[List[torch.Tensor], List[np.ndarray]]]:
        """Removes additional outputs and detections with zero score.

        Args:
            test_outputs (List[Union[torch.Tensor, np.ndarray]]):
                outputs of forward_test.

        Returns:
            List[Union[List[torch.Tensor], List[np.ndarray]]]:
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

    def __postprocessing_masks(self,
                               det_bboxes: np.ndarray,
                               det_masks: np.ndarray,
                               img_w: int,
                               img_h: int,
                               mask_thr_binary: float = 0.5) -> np.ndarray:
        """Additional processing of masks. Resizes masks from [num_det, 28, 28]
        to [num_det, img_w, img_h]. Analog of the 'mmdeploy.mmdet.models.roi_he
        ads.mask_heads.fcn_mask_head._do_paste_mask' function.

        Args:
            det_bboxes (np.ndarray): Bbox of shape [num_det, 5]
            det_masks (np.ndarray): Masks of shape [num_det, 28, 28].
            img_w (int): Width of the original image.
            img_h (int): Height of the original image.
            mask_thr_binary (float): The threshold for the mask.

        Returns:
            np.ndarray: masks of shape [N, num_det, img_w, img_h].
        """
        masks = det_masks
        bboxes = det_bboxes

        num_det = bboxes.shape[0]
        if num_det == 0:
            return np.zeros((0, img_w, img_h))

        if isinstance(masks, np.ndarray):
            masks = torch.tensor(masks)
            bboxes = torch.tensor(bboxes)

        result_masks = []
        for bbox, mask in zip(bboxes, masks):

            x0_int, y0_int = 0, 0
            x1_int, y1_int = img_w, img_h

            img_y = torch.arange(y0_int, y1_int, dtype=torch.float32) + 0.5
            img_x = torch.arange(x0_int, x1_int, dtype=torch.float32) + 0.5
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

            mask = img_masks
            mask = (mask >= mask_thr_binary).to(dtype=torch.bool)
            result_masks.append(mask.numpy())
        result_masks = np.concatenate(result_masks, axis=1)
        return result_masks.squeeze(0)

    def forward(self, img: Sequence[torch.Tensor], img_metas: Sequence[dict],
                *args, **kwargs):
        """Run forward inference.

        Args:
            img (Sequence[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of meta info for image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        input_img = img[0].contiguous()
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        outputs = self.__clear_outputs(outputs)
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = input_img.shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)

            dets_results = bbox2result(dets, labels, len(self.CLASSES))

            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                export_postprocess_mask = True
                if self.deploy_cfg is not None:
                    mmdet_deploy_cfg = get_mmdet_params(self.deploy_cfg)
                    # this flag enable postprocess when export.
                    export_postprocess_mask = mmdet_deploy_cfg.get(
                        'export_postprocess_mask', True)
                if not export_postprocess_mask:
                    masks = self.__postprocessing_masks(
                        dets[:, :4], masks, ori_w, ori_h)
                else:
                    masks = masks[:, :img_h, :img_w]
                # avoid to resize masks with zero dim
                if rescale and masks.shape[0] != 0:
                    masks = masks.astype(np.float32)
                    masks = torch.from_numpy(masks)
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0), size=(ori_h, ori_w))
                    masks = masks.squeeze(0).detach().numpy()
                if masks.dtype != np.bool:
                    masks = masks >= 0.5
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        return results


class ONNXRuntimeDetector(DeployBaseDetector):
    """Wrapper for detection's inference with ONNXRuntime.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 device_id: int, **kwargs):
        super(ONNXRuntimeDetector, self).__init__(class_names, device_id,
                                                  **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        ort_outputs = self.model({'input': imgs})
        return ort_outputs


class TensorRTDetector(DeployBaseDetector):
    """Wrapper for detection's inference with TensorRT.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 device_id: int, **kwargs):
        super(TensorRTDetector, self).__init__(class_names, device_id,
                                               **kwargs)
        from mmdeploy.apis.tensorrt import TRTWrapper

        self.model = TRTWrapper(model_file)
        self.output_names = ['dets', 'labels']
        if len(self.model.output_names) == 3:
            self.output_names.append('masks')

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model({'input': imgs})
            outputs = [outputs[name] for name in self.output_names]
        outputs = [out.detach().cpu().numpy() for out in outputs]
        # filtered out invalid output filled with -1
        batch_labels = outputs[1]
        batch_size = batch_labels.shape[0]
        inds = batch_labels.reshape(-1) != -1
        for i in range(len(outputs)):
            ori_shape = outputs[i].shape
            outputs[i] = outputs[i].reshape(-1,
                                            *ori_shape[2:])[inds, ...].reshape(
                                                batch_size, -1, *ori_shape[2:])
        return outputs


class PPLDetector(DeployBaseDetector):
    """Wrapper for detection's inference with PPL.

    Args:
        model_file (str): Path of input ONNX model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(PPLDetector, self).__init__(class_names, device_id)
        from mmdeploy.apis.ppl import PPLWrapper
        self.model = PPLWrapper(model_file, device_id)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class
                labels of shape [N, num_det].
        """
        ppl_outputs = self.model({'input': imgs})
        return ppl_outputs


class OpenVINODetector(DeployBaseDetector):
    """Wrapper for detector's inference with OpenVINO.

    Args:
        model_file (str): The path of input model file (.xml).
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 device_id: int, **kwargs):
        super(OpenVINODetector, self).__init__(class_names, device_id,
                                               **kwargs)
        from mmdeploy.apis.openvino import OpenVINOWrapper
        self.model = OpenVINOWrapper(model_file)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> Tuple:
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            If there are no masks in the output:
                tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5]
                    and class labels of shape [N, num_det].
            If the output contains masks:
                tuple[np.ndarray, np.ndarray, np.ndarray]:
                    dets of shape [N, num_det, 5],
                    class labels of shape [N, num_det] and
                    masks of shape [N, num_det, H, W].
        """
        openvino_outputs = self.model({'input': imgs})
        output_keys = ['dets', 'labels']
        if 'masks' in openvino_outputs:
            output_keys += ['masks']
        openvino_outputs = [openvino_outputs[key] for key in output_keys]
        return openvino_outputs


class PartitionSingleStageDetector(DeployBaseDetector):
    """Base wrapper for partitioned single stage detector.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(PartitionSingleStageDetector,
              self).__init__(class_names, device_id, **kwargs)
        # load cfg if necessary
        deploy_cfg = load_config(deploy_cfg)[0]
        model_cfg = load_config(model_cfg)[0]

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

    def partition0_postprocess(self, scores: torch.Tensor,
                               bboxes: torch.Tensor):
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

        post_params = get_mmdet_params(deploy_cfg)
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
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


class ONNXRuntimePSSDetector(PartitionSingleStageDetector):
    """Wrapper for partitioned single stage detector with ONNX Runtime.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(ONNXRuntimePSSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(
            model_file, device_id, output_names=['scores', 'boxes'])

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        ort_outputs = self.model({'input': imgs})
        scores, bboxes = ort_outputs[:2]
        scores = torch.from_numpy(scores).to(imgs.device)
        bboxes = torch.from_numpy(bboxes).to(imgs.device)
        return self.partition0_postprocess(scores, bboxes)


class TensorRTPSSDetector(PartitionSingleStageDetector):
    """TensorRT Wrapper for partition single stage detector.

    Args:
        model_file (str): Path of the engine file.
        class_names (list[str] | tuple[str]): Class names of the detector.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        deploy_cfg (str | mmcv.Config): Deployment config file or Config
            object.
        device_id (int): Device index, should be same as the engine.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(TensorRTPSSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)
        from mmdeploy.apis.tensorrt import TRTWrapper

        self.model = TRTWrapper(model_file)
        self.output_names = ['scores', 'boxes']

    def forward_test(self, imgs: torch.Tensor, *args,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward test.

        Args:
            imgs (torch.Tensor): The input image(s).

        Return:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model({'input': imgs})
            outputs = [outputs[name] for name in self.output_names]
        scores, bboxes = outputs[:2]
        return self.partition0_postprocess(scores, bboxes)


class NCNNPSSDetector(PartitionSingleStageDetector):
    """Wrapper for partitioned single stage detector with NCNN.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(NCNNPSSDetector, self).__init__(class_names, model_cfg,
                                              deploy_cfg, device_id, **kwargs)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert len(model_file) == 2
        ncnn_param_file = model_file[0]
        ncnn_bin_file = model_file[1]
        self.model = NCNNWrapper(
            ncnn_param_file, ncnn_bin_file, output_names=['boxes', 'scores'])

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run forward test.

        Args:
            imgs (torch.Tensor): The input image(s).

        Return:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        outputs = self.model({'input': imgs})
        boxes = outputs['boxes']
        scores = outputs['scores']
        return self.partition0_postprocess(scores, boxes)


class PartitionTwoStageDetector(DeployBaseDetector):
    """Base wrapper for partitioned two stage detector.

    Args:
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(PartitionTwoStageDetector,
              self).__init__(class_names, device_id, **kwargs)
        from mmdet.models.builder import build_head, build_roi_extractor

        from mmdeploy.mmdet.models.roi_heads.bbox_heads import \
            get_bboxes_of_bbox_head

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

        self.bbox_roi_extractor = build_roi_extractor(
            model_cfg.model.roi_head.bbox_roi_extractor)
        self.bbox_head = build_head(model_cfg.model.roi_head.bbox_head)

        class Context:
            pass

        ctx = Context()
        ctx.cfg = self.deploy_cfg
        self.get_bboxes_of_bbox_head = partial(get_bboxes_of_bbox_head, ctx)

    def partition0_postprocess(self, x: Sequence[torch.Tensor],
                               scores: torch.Tensor, bboxes: torch.Tensor):
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

        post_params = get_mmdet_params(deploy_cfg)
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
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

    def partition1_postprocess(self, rois: torch.Tensor,
                               cls_score: torch.Tensor,
                               bbox_pred: torch.Tensor,
                               img_metas: Sequence[dict]):
        """Perform post-processing for partition 1.
        Args:
            rois (torch.Tensor): Input tensor of roi.
            cls_score (torch.Tensor): Scores of all classes.
            bbox_pred (torch.Tensor): Bounding box proposals.
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
        return self.get_bboxes_of_bbox_head(self.bbox_head, rois, cls_score,
                                            bbox_pred,
                                            img_metas[0][0]['img_shape'],
                                            rcnn_test_cfg)


class ONNXRuntimePTSDetector(PartitionTwoStageDetector):
    """Wrapper for partitioned two stage detector with ONNX Runtime.

    Args:
        model_file (Sequence[str]): A list of paths of input model files.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: Sequence[str], class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(ONNXRuntimePTSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model_list = [
            ORTWrapper(file, device_id=device_id) for file in model_file
        ]
        num_partition0_outputs = len(self.model_list[0].output_names)
        num_feat = num_partition0_outputs - 2
        self.model_list[0].output_names = [
            'feat/{}'.format(i) for i in range(num_feat)
        ] + ['scores', 'boxes']
        self.model_list[1].output_names = ['cls_score', 'bbox_pred']

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of image(s) meta information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class
                labels of shape [N, num_det].
        """
        ort_outputs = self.model_list[0]({'input': imgs})
        feats = ort_outputs[:-2]
        scores, bboxes = ort_outputs[-2:]
        feats = [torch.from_numpy(feat).to(imgs.device) for feat in feats]
        scores = torch.from_numpy(scores).to(imgs.device)
        bboxes = torch.from_numpy(bboxes).to(imgs.device)

        # partition0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # partition1
        ort_outputs = self.model_list[1]({'bbox_feats': bbox_feats})
        cls_score, bbox_pred = ort_outputs[:2]
        cls_score = torch.from_numpy(cls_score).to(imgs.device)
        bbox_pred = torch.from_numpy(bbox_pred).to(imgs.device)

        # partition1_postprocess
        return self.partition1_postprocess(rois, cls_score, bbox_pred,
                                           img_metas)


class TensorRTPTSDetector(PartitionTwoStageDetector):
    """Wrapper for partitioned two stage detector with TensorRT.

    Args:
        model_file (Sequence[str]): A list of paths of input model files.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: Sequence[str], class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(TensorRTPTSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)

        from mmdeploy.apis.tensorrt import TRTWrapper

        model_list = []
        for m_file in model_file:
            model = TRTWrapper(m_file)
            model_list.append(model)

        self.model_list = model_list

        output_names_list = []
        num_partition0_outputs = len(model_list[0].output_names)
        num_feat = num_partition0_outputs - 2
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_feat)] + ['scores', 'boxes'])  # partition0
        output_names_list.append(['cls_score', 'bbox_pred'])  # partition1
        self.output_names_list = output_names_list

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of image(s) meta information.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[0]({'input': imgs})
            outputs = [outputs[name] for name in self.output_names_list[0]]
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # partition0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # partition1 forward
        bbox_feats = bbox_feats.contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[1]({'bbox_feats': bbox_feats})
            outputs = [outputs[name] for name in self.output_names_list[1]]
        cls_score, bbox_pred = outputs[:2]

        # partition1_postprocess
        outputs = self.partition1_postprocess(rois, cls_score, bbox_pred,
                                              img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


class NCNNPTSDetector(PartitionTwoStageDetector):
    """Wrapper for partitioned two stage detector with NCNN.

    Args:
        model_file (Sequence[str]): A list of paths of input model files.
        class_names (Sequence[str]): A list of string specifying class names.
        model_cfg: (str | mmcv.Config): Input model config.
        deploy_cfg: (str | mmcv.Config): Input deployment config.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: Sequence[str], class_names: Sequence[str],
                 model_cfg: Union[str, mmcv.Config],
                 deploy_cfg: Union[str,
                                   mmcv.Config], device_id: int, **kwargs):
        super(NCNNPTSDetector, self).__init__(class_names, model_cfg,
                                              deploy_cfg, device_id, **kwargs)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert self.device_id == -1
        assert len(model_file) == 4

        model_list = []
        for ncnn_param_file, ncnn_bin_file in zip(model_file[::2],
                                                  model_file[1::2]):
            model = NCNNWrapper(ncnn_param_file, ncnn_bin_file)
            model_list.append(model)

        model_cfg = load_config(model_cfg)[0]
        num_output_stage1 = model_cfg['model']['neck']['num_outs']

        output_names_list = []
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_output_stage1)] + ['scores', 'boxes'])
        output_names_list.append(['cls_score', 'bbox_pred'])

        model_list[0].set_output_names(output_names_list[0])
        model_list[1].set_output_names(output_names_list[1])

        self.model_list = model_list
        self.output_names_list = output_names_list

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     *args, **kwargs):
        """Implement forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]): A list of image(s) meta information.

        Returns:
            tuple[np.ndarray, np.ndarray]: dets of shape [N, num_det, 5] and
                class labels of shape [N, num_det].
        """
        # stage0 forward
        out_stage0 = self.model_list[0]({'input': imgs})

        outputs = []
        for name in self.output_names_list[0]:
            out = out_stage0[name]
            outputs.append(out)
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # stage0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # stage1 forward
        out_stage1 = self.model_list[1]({'bbox_feats': bbox_feats})
        cls_score = out_stage1['cls_score']
        bbox_pred = out_stage1['bbox_pred']

        # stage1_postprocess
        outputs = self.partition1_postprocess(rois, cls_score, bbox_pred,
                                              img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


def get_classes_from_config(model_cfg: Union[str, mmcv.Config], **kwargs):
    """Get class name from config.

    Args:
        model_cfg (str | mmcv.Config): Input model config file or
            Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    # load cfg if necessary
    model_cfg = load_config(model_cfg)[0]
    module_dict = DATASETS.module_dict
    data_cfg = model_cfg.data

    if 'test' in data_cfg:
        module = module_dict[data_cfg.test.type]
    elif 'val' in data_cfg:
        module = module_dict[data_cfg.val.type]
    elif 'train' in data_cfg:
        module = module_dict[data_cfg.train.type]
    else:
        raise RuntimeError(f'No dataset config found in: {model_cfg}')

    return module.CLASSES


ONNXRUNTIME_DETECTOR_MAP = dict(
    end2end=ONNXRuntimeDetector,
    single_stage=ONNXRuntimePSSDetector,
    two_stage=ONNXRuntimePTSDetector)

TENSORRT_DETECTOR_MAP = dict(
    end2end=TensorRTDetector,
    single_stage=TensorRTPSSDetector,
    two_stage=TensorRTPTSDetector)

PPL_DETECTOR_MAP = dict(end2end=PPLDetector)

NCNN_DETECTOR_MAP = dict(
    single_stage=NCNNPSSDetector, two_stage=NCNNPTSDetector)

OPENVINO_MAP = dict(end2end=OpenVINODetector)

BACKEND_DETECTOR_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_DETECTOR_MAP,
    Backend.TENSORRT: TENSORRT_DETECTOR_MAP,
    Backend.PPL: PPL_DETECTOR_MAP,
    Backend.NCNN: NCNN_DETECTOR_MAP,
    Backend.OPENVINO: OPENVINO_MAP
}


def build_detector(model_files: Sequence[str], model_cfg: Union[str,
                                                                mmcv.Config],
                   deploy_cfg: Union[str,
                                     mmcv.Config], device_id: int, **kwargs):
    """Build detector for different backend.

    Args:
        model_files (list[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device_id (int): An integer represents device index.

    Returns:
        DeployBaseDetector: Detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    class_names = get_classes_from_config(model_cfg)

    assert backend in BACKEND_DETECTOR_MAP, \
        f'Unsupported backend type: {backend.value}'
    detector_map = BACKEND_DETECTOR_MAP[backend]

    partition_type = 'end2end'
    partition_config = get_partition_config(deploy_cfg)
    if partition_config is not None:
        partition_type = partition_config.get('type', None)

    assert partition_type in detector_map,\
        f'Unsupported partition type: {partition_type}'
    backend_detector_class = detector_map[partition_type]

    model_files = model_files[0] if len(model_files) == 1 else model_files
    backend_detector = backend_detector_class(
        model_file=model_files,
        class_names=class_names,
        device_id=device_id,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_detector

# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.detectors import BaseDetector
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
        batch_inputs = inputs.contiguous()
        outputs = self.predict(batch_inputs)
        outputs = End2EndModel.__clear_outputs(outputs)
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = batch_inputs.shape[0]
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

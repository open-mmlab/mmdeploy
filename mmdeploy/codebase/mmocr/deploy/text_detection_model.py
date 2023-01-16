# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import cv2
import mmengine
import torch
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData
from mmocr.structures import TextDetDataSample

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

__BACKEND_MODEL = Registry('backend_text_detectors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of text detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (mmengine.Config | None): Loaded Config object of MMDeploy.
        model_cfg (mmengine.Config | None): Loaded Config object of MMOCR.
    """

    def __init__(
        self,
        backend: Backend,
        backend_files: Sequence[str],
        device: str,
        deploy_cfg: Optional[mmengine.Config] = None,
        model_cfg: Optional[mmengine.Config] = None,
        **kwargs,
    ):
        data_preprocessor = model_cfg.model.get('data_preprocessor', {})
        if data_preprocessor is not None:  # skip when it is SDKEnd2EndModel
            data_preprocessor.update(
                model_cfg.model.get('cfg', {}).get('data_preprocessor', {}))
            if data_preprocessor.get('type', None) == 'DetDataPreprocessor':
                data_preprocessor.update(_scope_='mmdet')  # MRCNN
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.show_score = False

        from mmocr.registry import MODELS
        if hasattr(model_cfg.model, 'det_head'):
            self.det_head = MODELS.build(model_cfg.model.det_head)
        else:
            self.text_repr_type = model_cfg.model.get('text_repr_type', 'poly')
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
        """Initialize the wrapper of backends.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. .onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string represents device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                **kwargs) -> Sequence[TextDetDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (List[BaseDataElement] | None): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            list[TextDetDataSample]: A list of N datasamples of prediction
            results.  Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - polygons (list[np.ndarray]): The length is num_instances.
                    Each element represents the polygon of the
                    instance, in (xn, yn) order.
        """
        x = self.extract_feat(inputs)
        if hasattr(self, 'det_head'):
            return self.det_head.postprocessor(x[0], data_samples)
        # post-process of mmdet models
        from mmdet.structures.mask import bitmap_to_polygon
        from mmocr.utils.bbox_utils import bbox2poly

        from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
        from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
            End2EndModel as DetModel
        if len(x) == 3:  # instance seg
            batch_dets, _, batch_masks = x
            for i in range(batch_dets.size(0)):
                masks = batch_masks[i]
                bboxes = batch_dets[i, :, :4]
                bboxes[:, ::2] /= data_samples[i].scale_factor[0]
                bboxes[:, 1::2] /= data_samples[i].scale_factor[1]
                ori_h, ori_w = data_samples[i].ori_shape[:2]
                img_h, img_w = data_samples[i].img_shape[:2]
                export_postprocess_mask = True
                polygons = []
                scores = []
                if self.deploy_cfg is not None:
                    codebase_cfg = get_post_processing_params(self.deploy_cfg)
                    # this flag enable postprocess when export.
                    export_postprocess_mask = codebase_cfg.get(
                        'export_postprocess_mask', True)
                if not export_postprocess_mask:
                    masks = DetModel.postprocessing_masks(
                        bboxes, masks, ori_w, ori_h, batch_masks.device)
                else:
                    masks = masks[:, :img_h, :img_w]
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0).float(), size=(ori_h, ori_w))
                masks = masks.squeeze(0)
                if masks.dtype != bool:
                    masks = masks >= 0.5

                for mask_idx, mask in enumerate(masks.cpu()):
                    contours, _ = bitmap_to_polygon(mask)
                    polygons += [contour.reshape(-1) for contour in contours]
                    scores += [batch_dets[i, :, 4][mask_idx].cpu()
                               ] * len(contours)
                # filter invalid polygons
                filterd_polygons = []
                keep_idx = []
                for poly_idx, polygon in enumerate(polygons):
                    if len(polygon) < 6:
                        continue
                    filterd_polygons.append(polygon)
                    keep_idx.append(poly_idx)
                # convert by text_repr_type
                if self.text_repr_type == 'quad':
                    for j, poly in enumerate(filterd_polygons):
                        rect = cv2.minAreaRect(poly)
                        vertices = cv2.boxPoints(rect)
                        poly = vertices.flatten()
                        filterd_polygons[j] = poly
                pred_instances = InstanceData()
                pred_instances.polygons = filterd_polygons
                pred_instances.scores = torch.FloatTensor(scores)[keep_idx]
                data_samples[i].pred_instances = pred_instances
        else:
            dets = x[0]
            for i in range(dets.size(0)):
                bboxes = dets[i, :, :4].cpu().numpy()
                bboxes[:, ::2] /= data_samples[i].scale_factor[0]
                bboxes[:, 1::2] /= data_samples[i].scale_factor[1]
                polygons = [bbox2poly(bbox) for bbox in bboxes]
                pred_instances = InstanceData()
                pred_instances.polygons = polygons
                pred_instances.scores = torch.FloatTensor(dets[i, :, 4].cpu())
                data_samples[i].pred_instances = pred_instances
        return data_samples

    def extract_feat(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """The interface for forward test.

        Args:
            batch_inputs (torch.Tensor): Input image(s) in
            [N x C x H x W] format.

        Returns:
            List[torch.Tensor]: A list of predictions of input images.
        """
        outputs = self.wrapper({self.input_name: batch_inputs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmocr format."""

    def __init__(self, *args, **kwargs):
        kwargs['model_cfg'].model.data_preprocessor = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                *args,
                **kwargs) -> list:
        """Run forward inference.

        Args:
            inputs (torch.Tensor): Images of shape (N, C, H, W).
            data_samples (List[BaseDataElement] | None): A list of N
                datasamples, containing meta information and gold annotations
                for each of the images.

        Returns:
            list: A list contains predictions.
        """
        boundaries = self.wrapper.invoke(inputs[0].permute(
            [1, 2, 0]).contiguous().detach().cpu().numpy())
        polygons = [boundary[:-1] for boundary in boundaries]
        scores = torch.Tensor([boundary[-1] for boundary in boundaries])
        boundaries = [list(x) for x in boundaries]
        pred_instances = InstanceData()
        pred_instances.polygons = polygons
        pred_instances.scores = scores
        data_samples[0].pred_instances = pred_instances
        return data_samples


def build_text_detection_model(model_files: Sequence[str],
                               model_cfg: Union[str, mmengine.Config],
                               deploy_cfg: Union[str, mmengine.Config],
                               device: str, **kwargs):
    """Build text detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmengine.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmengine.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Text detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_text_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg,
            **kwargs))
    backend_text_detector = backend_text_detector.to(device)

    return backend_text_detector

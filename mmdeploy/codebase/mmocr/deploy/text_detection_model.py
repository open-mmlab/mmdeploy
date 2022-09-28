# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

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
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg,
            data_preprocessor=model_cfg.model.data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.show_score = False

        from mmocr.registry import MODELS
        self.det_head = MODELS.build(model_cfg.model.det_head)
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
        return self.det_head.postprocessor(x[0], data_samples)

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

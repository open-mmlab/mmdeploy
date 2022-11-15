# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet.structures.bbox import scale_boxes
from mmengine import Config, Registry
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.structures import BaseDataElement, InstanceData
from mmrotate.structures.bbox import RotatedBoxes
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

__BACKEND_MODEL = Registry('backend_detectors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of rotated detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        class_names (Sequence[str]): A list of string specifying class names.
        device (str): A string represents device type.
        deploy_cfg (Config): Deployment config file or loaded
            Config object.
    """

    def __init__(
        self,
        backend: Backend,
        backend_files: Sequence[str],
        device: str,
        deploy_cfg: Config,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
    ):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
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
            deploy_cfg=self.deploy_cfg)

    @staticmethod
    def __clear_outputs(
        test_outputs: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[Union[List[torch.Tensor], List[np.ndarray]]]:
        """Removes additional outputs and detections with zero and negative
        score.

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
            inds = test_outputs[0][i, :, -1] > 0.0
            for output_id in range(num_outputs):
                outputs[output_id][i] = test_outputs[output_id][i, inds, ...]
        return outputs

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                **kwargs) -> Any:
        """Run forward inference.

        Args:
            img (Sequence[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', 'Deploy model only allow mode=="predict".'
        inputs = inputs.contiguous()
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        outputs = self.predict(inputs)
        outputs = End2EndModel.__clear_outputs(outputs)
        batch_dets, batch_labels = outputs[:2]
        batch_size = inputs.shape[0]
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            bboxes = RotatedBoxes(dets[:, :-1], clone=False)
            scores = dets[:, -1]
            result = InstanceData()

            if rescale:
                scale_factor = img_metas[i]['scale_factor']
                scale_factor = [1 / s for s in scale_factor]
                bboxes = scale_boxes(bboxes, scale_factor)

            result.scores = scores
            result.bboxes = bboxes
            result.labels = labels

            data_samples[i].pred_instances = result
            results.append(data_samples[i])

        return results

    def predict(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[torch.Tensor]: A list of predictions of input images.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmrotate format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        results = []
        dets, labels = self.wrapper.invoke(
            img[0].contiguous().detach().cpu().numpy())
        dets_results = [dets[labels == i, :] for i in range(len(self.CLASSES))]
        results.append(dets_results)

        return results


def build_rotated_detection_model(
        model_files: Sequence[str],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build rotated detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Rotated detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, = load_config(deploy_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_rotated_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_rotated_detector

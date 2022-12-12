# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmengine
import torch
from mmengine.registry import Registry
from mmengine.structures import LabelData
from mmocr.utils.typing_utils import RecSampleList

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

__BACKEND_MODEL = Registry('backend_text_recognizer')


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
    ):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg,
            data_preprocessor=model_cfg.model.data_preprocessor)
        model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.show_score = False

        from mmocr.registry import MODELS, TASK_UTILS
        decoder = model_cfg.model.decoder
        assert decoder is not None, 'model_cfg contains no label '
        'decoder'
        max_seq_len = 40  # default value in EncodeDecodeRecognizer of mmocr
        if decoder.get('max_seq_len', None) is None:
            decoder.update(max_seq_len=max_seq_len)
        self.dictionary = TASK_UTILS.build(model_cfg.dictionary)
        if decoder.get('dictionary', None) is None:
            decoder.update(dictionary=self.dictionary)
        self.decoder = MODELS.build(decoder)
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

    def forward(self, inputs: torch.Tensor, data_samples: RecSampleList, *args,
                **kwargs) -> RecSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        out_enc = self.extract_feat(inputs)
        return self.decoder.postprocessor(out_enc, data_samples)

    def extract_feat(self, imgs: torch.Tensor, *args,
                     **kwargs) -> torch.Tensor:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Image input tensor.

        Returns:
            list[str]: Text label result of each image.
        """
        pred = self.wrapper({self.input_name: imgs})['output']
        return pred


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmocr format."""

    def __init__(self, *args, **kwargs):
        kwargs['model_cfg'].model.data_preprocessor = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)

    def forward(self, inputs: Sequence[torch.Tensor],
                data_samples: RecSampleList, *args, **kwargs):
        """Run forward inference.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[str]: Text label result of each image.
        """
        text, score = self.wrapper.invoke(inputs[0].permute(
            [1, 2, 0]).contiguous().detach().cpu().numpy())
        pred_text = LabelData()
        pred_text.score = score
        pred_text.item = text
        data_samples[0].pred_text = pred_text
        return data_samples


def build_text_recognition_model(model_files: Sequence[str],
                                 model_cfg: Union[str, mmengine.Config],
                                 deploy_cfg: Union[str, mmengine.Config],
                                 device: str, **kwargs):
    """Build text recognition model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmengine.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmengine.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Text recognizer for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_text_recognizer = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg,
            **kwargs))
    backend_text_recognizer = backend_text_recognizer.to(device)

    return backend_text_recognizer

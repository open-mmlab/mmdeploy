# Copyright (c) OpenMMLab. All rights reserved.
import mmocr.utils as utils
import numpy as np
import torch
import torch.nn as nn

from mmdeploy.core import MODULE_REWRITER
from mmdeploy.utils import is_dynamic_shape
from ..utils import get_resize_ocr


@MODULE_REWRITER.register_rewrite_module(
    'mmocr.models.textrecog.recognizer.SARNet', backend='default')
class SARNet(nn.Module):
    """SARNet network structure for OCR image recognition.

    SARNet apply an argument named `valid_ratio` in network structure and this
    specific argument can not be ignored or replaced during exporting to onnx.
    Hence, some information of model config are utilized to compute
    `valid_ratio`.

    Paper: Show, attend and read: A simple and strong baseline for irregular
    text recognition.
    """

    def __init__(self, module, deploy_cfg, **kwargs):
        super(SARNet, self).__init__()

        self._module = module
        self.deploy_cfg = deploy_cfg

    def forward(self, img, *args, **kwargs):
        """Run forward."""
        img_metas = [{}]
        assert utils.is_type_list(img_metas, dict)
        assert isinstance(img, torch.Tensor)

        is_dynamic_flag = is_dynamic_shape(self.deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]['img_shape'] = img_shape

        # compute valid_ratio through information in model config
        min_width, max_width, kepp_aspect_ratio = get_resize_ocr(
            self._module.cfg)
        valid_ratio = torch.tensor(1.0, device=img.device)
        if kepp_aspect_ratio:
            valid_ratio = 1 - (img == -1).sum() / np.array(img.size()).prod()
        valid_ratio = torch.clamp(valid_ratio, min_width / max_width, 1.0)
        img_metas[0]['valid_ratio'] = valid_ratio
        return self._module.simple_test(img, img_metas, *args, **kwargs)

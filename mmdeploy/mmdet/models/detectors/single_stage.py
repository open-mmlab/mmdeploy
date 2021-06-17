import torch
import torch.nn as nn

from mmdeploy.utils import MODULE_REWRITERS


@MODULE_REWRITERS.register_rewrite_module(module_type='mmdet.models.RetinaNet')
@MODULE_REWRITERS.register_rewrite_module(module_type='mmdet.models.SingleStageDetector')
class SingleStageDetector(nn.Module):
    def __init__(self, module, cfg, **kwargs):
        super(SingleStageDetector, self).__init__()
        self.module = module
        self.backbone = module.backbone
        self.with_neck = module.with_neck
        if module.neck is not None:
            self.neck = module.neck

        self.bbox_head = module.bbox_head

    def forward(self, data, **kwargs):
        # get origin input shape to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(data)[2:]
        x = self.module.extract_feat(data)
        outs = self.bbox_head(x)
        return self.bbox_head.get_bboxes(*outs, img_shape, **kwargs)

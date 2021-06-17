from mmdeploy.utils import MODULE_REWRITERS
from torch import nn


@MODULE_REWRITERS.register_rewrite_module(
    'mmcls.models.classifiers.ImageClassifier', backend='default')
@MODULE_REWRITERS.register_rewrite_module(
    'mmcls.models.classifiers.BaseClassifier', backend='default')
class BaseClassifierWrapper(nn.Module):

    def __init__(self, module, cfg, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, img, *args, **kwargs):
        return self.module.simple_test(img, {})

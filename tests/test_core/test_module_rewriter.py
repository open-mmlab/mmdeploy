# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import MODULE_REWRITER, patch_model


def test_module_rewriter():
    from torchvision.models.resnet import resnet50

    @MODULE_REWRITER.register_rewrite_module(
        module_type='torchvision.models.resnet.Bottleneck', backend='tensorrt')
    class BottleneckWrapper(torch.nn.Module):

        def __init__(self, module, cfg, **kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs) * 2

    x = torch.rand(1, 64, 32, 32)
    model = resnet50().eval()
    bottle_neck = model.layer1[0]
    result = bottle_neck(x)

    # rewrite module
    cfg = dict()

    rewritten_model = patch_model(model, cfg=cfg, backend='tensorrt')
    rewritten_bottle_nect = rewritten_model.layer1[0]
    rewritten_result = rewritten_bottle_nect(x)
    torch.testing.assert_allclose(rewritten_result, result * 2)

    # wrong backend should not be rewritten
    model = resnet50().eval()
    bottle_neck = model.layer1[0]
    result = bottle_neck(x)
    rewritten_model = patch_model(model, cfg=cfg)
    rewritten_bottle_nect = rewritten_model.layer1[0]
    rewritten_result = rewritten_bottle_nect(x)
    torch.testing.assert_allclose(rewritten_result, result)


def test_pass_redundant_args_to_model():
    from torchvision.models.resnet import resnet50

    @MODULE_REWRITER.register_rewrite_module(
        module_type='torchvision.models.resnet.Bottleneck')
    class BottleneckWrapper(torch.nn.Module):

        def __init__(self, module, cfg):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs) * 2

    model = resnet50().eval()

    rewritten_model = patch_model(model, cfg={}, redundant_args=12345)
    assert rewritten_model is not None

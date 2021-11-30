# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdeploy.codebase.mmseg.deploy import convert_syncbatchnorm


def test_convert_syncbatchnorm():

    class ExampleModel(nn.Module):

        def __init__(self):
            super(ExampleModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(2, 4), nn.SyncBatchNorm(4), nn.Sigmoid(),
                nn.Linear(4, 6), nn.SyncBatchNorm(6), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    model = ExampleModel()
    out_model = convert_syncbatchnorm(model)
    assert isinstance(out_model.model[1],
                      torch.nn.modules.batchnorm.BatchNorm2d) and isinstance(
                          out_model.model[4],
                          torch.nn.modules.batchnorm.BatchNorm2d)

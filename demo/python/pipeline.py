# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import cv2
from mmdeploy_python import Pipeline, Model, Context, Device, Foo, test_foo

foo = Foo()
print(test_foo(foo))
print(test_foo(0))

det_model = Model('/workspace/mmdeploy/benchmark/_detection_tmp_model')
reg_model = Model('/workspace/mmdeploy/benchmark/_mmcls_tmp_model')

config = dict(
    type='Pipeline',
    input='img',
    tasks=[
        dict(
            type='Inference',
            input='img',
            output='dets',
            params=dict(model=det_model)),
        dict(
            type='Pipeline',
            input=['boxes=*dets', 'imgs=+img'],
            tasks=[
                dict(
                    type='Task',
                    module='CropBox',
                    input=['imgs', 'boxes'],
                    output='patches'),
                dict(
                    type='Inference',
                    input='patches',
                    output='labels',
                    params=dict(model=reg_model))
            ],
            output='*labels')
    ],
    output=['dets', 'labels'])

device = Device('cuda')

pipeline = Pipeline(config, Context(device))

img = cv2.imread('/workspace/mmdetection/demo/demo.jpg')

output = pipeline(dict(ori_img=img))

print(output)
# print(json.dumps(output, indent=2))

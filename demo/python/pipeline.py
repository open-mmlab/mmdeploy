# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import cv2
from mmdeploy_python import Pipeline

config = dict(
    type='Pipeline',
    input='img',
    tasks=[
        dict(
            type='Inference',
            input='img',
            output='dets',
            params=dict(
                model=
                '/workspace/deploy_prototype/benchmark/_detection_tmp_model')),
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
                    params=dict(
                        model=
                        '/workspace/deploy_prototype/benchmark/_mmcls_tmp_model'
                    ))
            ],
            output='*labels')
    ],
    output=['dets', 'labels'])

pipeline = Pipeline(config, 'cuda', 0)

img = cv2.imread('/workspace/mmdetection/demo/demo.jpg')

output = pipeline([[dict(ori_img=img)]])

print(json.dumps(output, indent=2))

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import cv2
from mmdeploy_python import Context, Device, Model, Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo of MMDeploy SDK pipeline API')
    parser.add_argument('device', help='name of device, cuda or cpu')
    parser.add_argument('det_model_path', help='path of detection model')
    parser.add_argument('cls_model_path', help='path of classification model')
    parser.add_argument('image_path', help='path to test image')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    det_model = Model(args.det_model_path)
    reg_model = Model(args.cls_model_path)

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
                # flatten dets ([[a]] -> [a]) and broadcast img
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
                # unflatten labels ([a] -> [[a]])
                output='*labels')
        ],
        output=['dets', 'labels'])

    device = Device(args.device)
    pipeline = Pipeline(config, Context(device))

    img = cv2.imread(args.image_path)

    output = pipeline(dict(ori_img=img))

    print(json.dumps(output, indent=4))


if __name__ == '__main__':
    main()

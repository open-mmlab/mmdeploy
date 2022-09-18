# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from math import cos, sin

import cv2
import numpy as np
from mmdeploy_python import RotatedDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path', help='path of SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of an image')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)
    detector = RotatedDetector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    rbboxes, labels = detector(img)

    indices = [i for i in range(len(rbboxes))]
    for index, rbbox, label_id in zip(indices, rbboxes, labels):
        [cx, cy, w, h, angle], score = rbbox[0:5], rbbox[-1]
        if score < 0.1:
            continue
        [wx, wy, hx, hy] = \
            0.5 * np.array([w, w, -h, h]) * \
            np.array([cos(angle), sin(angle), sin(angle), cos(angle)])
        points = np.array([[[int(cx - wx - hx),
                             int(cy - wy - hy)],
                            [int(cx + wx - hx),
                             int(cy + wy - hy)],
                            [int(cx + wx + hx),
                             int(cy + wy + hy)],
                            [int(cx - wx + hx),
                             int(cy - wy + hy)]]])
        cv2.drawContours(img, points, -1, (0, 255, 0), 2)

    cv2.imwrite('output_detection.png', img)


if __name__ == '__main__':
    main()

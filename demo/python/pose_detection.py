# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_python import PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument(
        'model_path', help='the directory path of mmdeploy model')
    parser.add_argument('image_path', help='the path of an image')
    parser.add_argument(
        '--bbox',
        default=None,
        nargs='+',
        help='bounding box of an object in format (x, y, w, h)')
    parser.add_argument(
        '--device-name', default='cpu', help='the name of device, cuda or cpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    bboxes = []
    if args.bbox is None:
        bbox = [0, 0, img.shape[1], img.shape[0]]
    else:
        # x, y, w, h -> left, top, right, bottom
        bbox = np.array(args.bbox, dtype=int)
        bbox[2:] += bbox[:2]
    bboxes.append(bbox)

    detector = PoseDetector(args.model_path, args.device_name, 0)
    result = detector([img], [bboxes])[0]

    _, point_num, _ = result.shape
    points = result[:, :, :2].reshape(point_num, 2)
    for [x, y] in points.astype(int):
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    cv2.imwrite('output_pose.png', img)


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_runtime import PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of an image')
    parser.add_argument(
        '--bbox',
        default=None,
        nargs='+',
        type=int,
        help='bounding box of an object in format (x, y, w, h)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    detector = PoseDetector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)

    if args.bbox is None:
        result = detector(img)
    else:
        # converter (x, y, w, h) -> (left, top, right, bottom)
        print(args.bbox)
        bbox = np.array(args.bbox, dtype=int)
        bbox[2:] += bbox[:2]
        result = detector(img, bbox)
    dets, points = result
    dets = dets.reshape(-1, 5)
    for box in dets:
        score = box[4]
        x1, y1, x2, y2 = [int(_) for _ in box[:4]]
        if score < 0.3:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

    points = points[..., :2].reshape(-1, 2).astype(np.int32)
    for [x, y] in points:
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    cv2.imwrite('output_pose.png', img)


if __name__ == '__main__':
    main()

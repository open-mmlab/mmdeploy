# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_python import Detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument(
        'model_path', help='the directory path of mmdeploy model')
    parser.add_argument('image_path', help='the path of an image')
    parser.add_argument(
        '--device-name', default='cpu', help='the name of device, cuda or cpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)
    detector = Detector(args.model_path, args.device_name, 0)
    bboxes, labels, masks = detector([img])[0]
    assert (isinstance(bboxes, np.ndarray))
    assert (isinstance(labels, np.ndarray))
    assert (isinstance(masks, list))

    indices = [i for i in range(len(bboxes))]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.3:
            continue

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

        if masks[index].size:
            mask = masks[index]
            blue, green, red = cv2.split(img)
            mask_img = blue[top:top + mask.shape[0], left:left + mask.shape[1]]
            cv2.bitwise_or(mask, mask_img, mask_img)
            img = cv2.merge([blue, green, red])

    cv2.imwrite('output_detection.png', img)


if __name__ == '__main__':
    main()

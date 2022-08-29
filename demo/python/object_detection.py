# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_python import Detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of an image')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)
    detector = Detector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    bboxes, labels, masks = detector(img)

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

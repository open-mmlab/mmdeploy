# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_python import TextDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument(
        '--textdet',
        default='',
        help='the directory path of mmdeploy text-detector sdk model')
    parser.add_argument(
        '--textrecog',
        default='',
        help='the directory path of mmdeploy text-recognizer sdk model')
    parser.add_argument('image_path', help='the path of an image')
    parser.add_argument(
        '--device-name', default='cpu', help='the name of device, cuda or cpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    if args.textdet:
        detector = TextDetector(args.textdet, args.device_name, 0)
        bboxes = detector([img])[0]

        pts = (bboxes[:, 0:8] + 0.5).reshape(len(bboxes), -1, 2).astype(int)
        cv2.polylines(img, pts, True, (0, 255, 0), 2)
        cv2.imwrite('output_ocr.png', img)

    if args.textrecog:
        print('API of TextRecognizer does not support bbox as argument yet')


if __name__ == '__main__':
    main()

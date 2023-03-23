# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_runtime import TextDetector, TextRecognizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument('image_path', help='path of an image')
    parser.add_argument(
        '--textdet',
        default='',
        help='path of mmdeploy text-detector SDK model dumped by'
        'model converter',
    )
    parser.add_argument(
        '--textrecog',
        default='',
        help='path of mmdeploy text-recognizer SDK model dumped by'
        'model converter',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    if args.textdet:
        detector = TextDetector(
            model_path=args.textdet, device_name=args.device_name, device_id=0)
        bboxes = detector(img)
        print(f'bboxes.shape={bboxes.shape}')
        print(f'bboxes={bboxes}')
        if len(bboxes) > 0:
            pts = ((bboxes[:, 0:8] + 0.5).reshape(len(bboxes), -1,
                                                  2).astype(int))
            cv2.polylines(img, pts, True, (0, 255, 0), 2)
            cv2.imwrite('output_ocr.png', img)

        if len(bboxes) > 0 and args.textrecog:
            recognizer = TextRecognizer(
                model_path=args.textrecog,
                device_name=args.device_name,
                device_id=0,
            )
            texts = recognizer(img, bboxes.flatten().tolist())
            print(texts)

    elif args.textrecog:
        recognizer = TextRecognizer(
            model_path=args.textrecog,
            device_name=args.device_name,
            device_id=0,
        )
        texts = recognizer(img)
        print(texts)
    else:
        print('do nothing since neither text detection sdk model or '
              'text recognition sdk model in input')


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_python import Classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='the name of device, cuda or cpu')
    parser.add_argument(
        'model_path', help='the directory path of mmdeploy model')
    parser.add_argument('image_path', help='the path of an image')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)
    classifier = Classifier(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    result = classifier([img])
    for label_id, score in result[0]:
        print(label_id, score)


if __name__ == '__main__':
    main()

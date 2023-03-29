# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_runtime import Restorer


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

    restorer = Restorer(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    result = restorer(img)

    # convert to BGR
    result = result[..., ::-1]
    cv2.imwrite('output_restorer.bmp', result)


if __name__ == '__main__':
    main()

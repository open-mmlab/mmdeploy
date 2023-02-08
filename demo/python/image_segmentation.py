# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_python import Segmentor


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


def get_palette(num_classes=256):
    state = np.random.get_state()
    # random color
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3))
    np.random.set_state(state)
    return [tuple(c) for c in palette]


def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    segmentor = Segmentor(
        model_path=args.model_path, device_name=args.device_name, device_id=0)
    seg = segmentor(img)
    if seg.dtype == np.float32:
        seg = np.argmax(seg, axis=0)

    palette = get_palette()
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    cv2.imwrite('output_segmentation.png', img)


if __name__ == '__main__':
    main()

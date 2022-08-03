# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser

import webcam_demo

# from webcam_demo.misc import is_image, is_video


def parse_args():
    parser = ArgumentParser('')

    parser.add_argument('model_path')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--camera', type=str, default='0')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--fps', type=str, default='30')

    parser.add_argument('--skip', type=str, default='2')

    parser.add_argument('--output', type=str, default='')

    return parser.parse_args()


def main():

    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    file = ''

    try:
        camera_id = int(args.camera)
    except ValueError:
        camera_id = -1
        file = args.camera.lower()

    if camera_id != -1:
        demo = webcam_demo.WebcamDemo(camera_id, args.model_path,
                                      args.device, 0.5, int(args.fps),
                                      int(args.skip), args.output)
    elif webcam_demo.misc.is_image(file):
        demo = webcam_demo.ImageDemo(file, args.model_path, args.device,
                                     args.output)
    elif webcam_demo.misc.is_video(file):
        demo = webcam_demo.VideoDemo(file, args.model_path, args.device,
                                     args.output)
    else:
        raise NotImplementedError('File type not supported')

    demo.run()


if __name__ == '__main__':
    main()

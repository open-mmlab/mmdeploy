# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser

import src

# from webcam_demo.misc import is_image, is_video


def parse_args():
    parser = ArgumentParser('')

    parser.add_argument('pose_model_path')

    parser.add_argument('--detect', type=str, default='')

    parser.add_argument('--camera', type=str, default='0')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--fps', type=str, default='30')

    parser.add_argument('--skip', type=str, default='2')

    parser.add_argument('--output', type=str, default='')

    parser.add_argument('--code', type=str, default='XVID')

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    setattr(args, 'detect_model_path', args.detect)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    file = ''
    # frames = []

    try:
        camera_id = int(args.camera)
    except ValueError:
        camera_id = -1
        file = args.camera.lower()

    if camera_id != -1:
        # print('test')

        demo = src.WebcamDemo(camera_id, args.detect_model_path,
                              args.pose_model_path, args.device, 0.5,
                              int(args.fps), int(args.skip), args.output)
    elif src.misc.is_image(file):
        demo = src.ImageDemo(file, args.detect_model_path,
                             args.pose_model_path, args.device, args.output)
    elif src.misc.is_video(file):
        demo = src.VideoDemo(file, args.detect_model_path,
                             args.pose_model_path, args.device, args.output,
                             args.code)
    else:
        raise NotImplementedError('File type not supported')

    demo.run()


if __name__ == '__main__':
    main()

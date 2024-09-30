# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os.path as osp

import mmcv
import mmengine
import numpy as np

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, get_root_logger, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model inference visualization.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        required=True,
        help='deploy model path')
    parser.add_argument(
        '--checkpoint', default=None, help='model checkpoint path')
    parser.add_argument(
        '--device', help='device type for inference', default='cpu')
    parser.add_argument(
        '--test-img',
        type=str,
        nargs='+',
        required=True,
        help='image used to test model')
    parser.add_argument(
        '--batch',
        type=int,
        choices=[1, 2],
        help='batch size for inference, accepts only 1 or 2')
    parser.add_argument(
        '--save-dir',
        default='workdir',
        help='the dir to save inference results')
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    # load cfgs
    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    input_shape = get_input_shape(deploy_cfg)
    backend_model = task_processor.build_backend_model(
        args.model,
        data_preprocessor_updater=task_processor.update_data_preprocessor)
    torch_model = None
    if args.checkpoint is not None:
        torch_model = task_processor.build_pytorch_model(args.checkpoint)

    mmengine.mkdir_or_exist(args.save_dir)
    # get visualizer
    visualizer = task_processor.get_visualizer('mmdeploy', args.save_dir)

    for i in range(0, len(args.test_img), args.batch):
        imgs = args.test_img[i:(i + args.batch)]
        model_inputs, _ = task_processor.create_input(
            imgs,
            input_shape,
            data_preprocessor=getattr(backend_model, 'data_preprocessor',
                                      None))
        backend_results = backend_model.test_step(model_inputs)
        torch_results = [None] * len(imgs)
        if torch_model is not None:
            torch_results = torch_model.test_step(model_inputs)

        # get visualized results
        for img_path, torch_res, backend_res in zip(imgs, torch_results,
                                                    backend_results):
            _, filename = osp.split(img_path)
            output_file = osp.join(args.save_dir, filename)
            image = mmcv.imread(img_path, channel_order='rgb')
            visualizer.add_datasample(
                filename,
                image,
                data_sample=backend_res,
                draw_gt=False,
                show=False,
                out_file=None)
            drawn_img = visualizer.get_image()
            if torch_res:
                visualizer.add_datasample(
                    filename,
                    image,
                    data_sample=torch_res,
                    draw_gt=False,
                    show=False,
                    out_file=None)
                drawn_img_torch = visualizer.get_image()
                shape = drawn_img.shape
                dummy_img = np.full((shape[0], 20, shape[2]),
                                    255,
                                    dtype=np.uint8)
                drawn_img = np.concatenate(
                    (drawn_img, dummy_img, drawn_img_torch), axis=1)
            if args.show:
                visualizer.show(drawn_img, win_name=filename, wait_time=0)
            drawn_img = mmcv.image.rgb2bgr(drawn_img)
            mmcv.imwrite(drawn_img, output_file)
            logger.info(f'Saved to {output_file}')


if __name__ == '__main__':
    main()

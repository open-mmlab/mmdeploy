# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp

import numpy as np
import torch
from mmengine import DictAction
from prettytable import PrettyTable

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_root_logger
from mmdeploy.utils.config_utils import (Backend, get_backend, get_input_shape,
                                         load_config)
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy Model Latency Test Tool.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument('image_dir', help='Input directory to image files')
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument(
        '--device', help='device type for inference', default='cuda:0')
    parser.add_argument(
        '--shape',
        type=str,
        help='Input shape to test in `HxW` format, e.g., `800x1344`',
        default=None)
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup iterations before counting inference latency.',
        default=10)
    parser.add_argument(
        '--num-iter',
        type=int,
        help='Number of iterations to run the inference.',
        default=100)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def get_images(
        image_dir,
        extensions=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
    images = []
    files = glob.glob(osp.join(image_dir, '**', '*'), recursive=True)
    for f in files:
        _, ext = osp.splitext(f)
        if ext.lower() in extensions:
            images.append(f)
    return images


class TorchWrapper(torch.nn.Module):

    def __init__(self, model):
        super(TorchWrapper, self).__init__()

        self.model = model

    @TimeCounter.count_time(Backend.PYTORCH.value)
    def forward(self, *args, **kwargs):
        return self.model(*args, return_loss=False, **kwargs)


def main():
    args = parse_args()
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    logger = get_root_logger()
    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # merge options for model cfg
    if args.cfg_options is not None:
        model_cfg.merge_from_dict(args.cfg_options)

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    if args.shape is not None:
        h, w = [int(_) for _ in args.shape.split('x')]
        input_shape = [w, h]
    else:
        input_shape = get_input_shape(deploy_cfg)
        assert input_shape is not None, 'Input_shape should not be None'

    # create model an inputs
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)

    model = task_processor.build_backend_model(args.model)
    backend = get_backend(deploy_cfg).value
    model = model.eval().to(args.device)
    is_device_cpu = args.device == 'cpu'
    with_sync = not is_device_cpu
    if not is_device_cpu:
        torch.backends.cudnn.benchmark = True
    image_files = get_images(args.image_dir)
    nrof_image = len(image_files)
    assert nrof_image > 0, f'No image files found in {args.image_dir}'
    logger.info(f'Found totally {nrof_image} image files in {args.image_dir}')
    total_iters = args.num_iter + args.warmup
    if nrof_image < total_iters:
        np.random.seed(1234)
        image_files += [
            image_files[i]
            for i in np.random.choice(nrof_image, total_iters - nrof_image)
        ]
    image_files = image_files[:total_iters]
    with TimeCounter.activate(
            warmup=args.warmup, log_interval=20, with_sync=with_sync):
        for image in image_files:
            data, _ = task_processor.create_input(image, input_shape)
            model(data['inputs'].unsqueeze(0).to(args.device),
                  [data['data_sample']])

    print('----- Settings:')
    settings = PrettyTable()
    settings.header = False
    batch_size = 1
    settings.add_row(['batch size', batch_size])
    settings.add_row(['shape', f'{input_shape[1]}x{input_shape[0]}'])
    settings.add_row(['iterations', args.num_iter])
    settings.add_row(['warmup', args.warmup])
    print(settings)
    print('----- Results:')
    TimeCounter.print_stats(backend)


if __name__ == '__main__':
    main()

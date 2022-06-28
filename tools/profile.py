# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import torch
from mmcv import DictAction
from prettytable import PrettyTable

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import (Backend, get_backend, get_input_shape,
                                         load_config)
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy Model Latency Test Tool.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument(
        '--img', type=str, help='Test image files.', required=True)
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument(
        '--device', help='device used for conversion', default='cuda:0')
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
    data, _ = task_processor.create_input(args.img, input_shape)
    model_ext = osp.splitext(args.model[0])[1]
    is_pytorch = model_ext in ['.pth', '.pt']
    if is_pytorch:
        # load pytorch model
        model = task_processor.init_pytorch_model(args.model[0])
        model = TorchWrapper(model)
        backend = Backend.PYTORCH.value
    else:
        # load the model of the backend
        model = task_processor.init_backend_model(args.model)
        backend = get_backend(deploy_cfg).value

    model = model.eval().to(args.device)
    is_device_cpu = args.device == 'cpu'
    with_sync = not is_device_cpu
    if not is_device_cpu:
        torch.backends.cudnn.benchmark = True

    with TimeCounter.activate(
            warmup=args.warmup, log_interval=20, with_sync=with_sync):
        for _ in range(args.num_iter + args.warmup):
            model(**data)

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

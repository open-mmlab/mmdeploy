import argparse
import logging
import os.path as osp

import mmcv
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import torch2onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backend.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument(
        'img', help='image used to convert model and test model')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_start_method('spawn')

    deploy_cfg = args.deploy_cfg
    model_cfg = args.model_cfg
    checkpoint = args.checkpoint

    # create work_dir if not
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    ret_value = mp.Value('d', 0, lock=False)

    # convert model
    logging.info('start torch2onnx conversion.')
    process = Process(
        target=torch2onnx,
        args=(args.img, args.work_dir, deploy_cfg, model_cfg, checkpoint),
        kwargs=dict(device=args.device, ret_value=ret_value))
    process.start()
    process.join()

    if ret_value.value != 0:
        logging.error('torch2onnx failed.')
        exit()
    else:
        logging.info('torch2onnx success.')


if __name__ == '__main__':
    main()

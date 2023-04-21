# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import time

import mmcv
import mmengine
import numpy as np
from tqdm import tqdm

from mmdeploy.apis import visualize_model
from mmdeploy.utils import Backend, get_backend, get_root_logger, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model inference visualization.')
    parser.add_argument('--deploy-cfg', help='deploy config path')
    parser.add_argument('--model-cfg', help='model config path')
    parser.add_argument(
        '--deploy-path', type=str, nargs='+', help='deploy model path')
    parser.add_argument(
        '--checkpoint', default=None, help='model checkpoint path')
    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')
    parser.add_argument(
        '--save-dir',
        default=os.getcwd(),
        help='the dir to save inference results')
    parser.add_argument('--device', help='device to run model', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    deploy_model_path = args.deploy_path
    if not isinstance(deploy_model_path, list):
        deploy_model_path = [deploy_model_path]

    # load deploy_cfg
    deploy_cfg = load_config(deploy_cfg_path)[0]

    # create save_dir or generate default save_dir
    current_time = time.localtime()
    save_dir = osp.join(os.getcwd(),
                        time.strftime('%Y_%m_%d_%H_%M_%S', current_time))
    mmengine.mkdir_or_exist(save_dir)

    # get backend info
    backend = get_backend(deploy_cfg)
    extra = dict()
    if backend == Backend.SNPE:
        extra['uri'] = args.uri

    # iterate single_img
    for single_img in tqdm(args.test_img):
        filename = osp.basename(single_img)
        output_file = osp.join(save_dir, filename)
        visualize_model(model_cfg_path, deploy_cfg_path, deploy_model_path,
                        single_img, args.device, backend, output_file, False,
                        **extra)

        if checkpoint_path:
            pytorch_output_file = osp.join(save_dir, 'pytorch_out.jpg')
            visualize_model(model_cfg_path, deploy_cfg_path, [checkpoint_path],
                            single_img, args.device, Backend.PYTORCH,
                            pytorch_output_file, False)

            # concat pytorch result and backend result
            backend_result = mmcv.imread(output_file)
            pytorch_result = mmcv.imread(pytorch_output_file)
            result = np.concatenate((backend_result, pytorch_result), axis=1)
            mmcv.imwrite(result, output_file)

            # remove temp pytorch result
            os.remove(osp.join(save_dir, pytorch_output_file))


if __name__ == '__main__':
    main()

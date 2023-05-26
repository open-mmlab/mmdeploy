# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import time
from typing import Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch
from tqdm import tqdm

from mmdeploy.utils import (Backend, get_backend, get_input_shape,
                            get_root_logger, load_config)


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
        '--batch',
        type=int,
        choices=[1, 2],
        help='batch size for inference, accepts only 1 or 2')
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


def batch_visualize_model(
        model_cfg: Union[str, mmengine.Config],
        deploy_cfg: Union[str, mmengine.Config],
        imgs: Union[str, np.ndarray, Sequence[str]],
        device: str,
        backend_model_path: Union[str, Sequence[str]],
        checkpoint_path: Optional[Union[str, Sequence[str]]] = None,
        backend: Optional[Backend] = None,
        output_file: Optional[str] = None):
    """Run inference with PyTorch or backend model and show results.

    Args:
        model_cfg (str | mmengine.Config): Model config file or Config object.
        deploy_cfg (str | mmengine.Config): Deployment config file or Config
            object.
        img (str | np.ndarray | Sequence[str]): Input image file or numpy array
            for inference.
        device (str): A string specifying device type.
        backend_model_path (str | Sequence[str]): Input backend model or
            file(s).
        checkpoint_path (str | Sequence[str]): Input pytorch checkpoint
            model or file(s), defaults to `None`.
        backend (Backend): Specifying backend type, defaults to `None`.
        output_file (str): Output file to save visualized image, defaults to
            `None`. Only valid if `show_result` is set to `False`.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    from mmdeploy.apis.utils import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    input_shape = get_input_shape(deploy_cfg)
    if backend is None:
        backend = get_backend(deploy_cfg)

    if isinstance(backend_model_path, str):
        backend_model_path = [backend_model_path]

    assert len(
        backend_model_path) > 0, 'Model should have at least one element.'

    # build model
    if checkpoint_path is not None:
        pytorch_model = task_processor.build_pytorch_model(checkpoint_path)
    backend_model = task_processor.build_backend_model(
        backend_model_path,
        data_preprocessor_updater=task_processor.update_data_preprocessor)

    if isinstance(imgs, str) or not isinstance(imgs, Sequence):
        imgs = [imgs]

    # batch inference
    for batch_img in imgs:
        model_inputs, _ = task_processor.create_input(batch_img, input_shape)
        with torch.no_grad():
            if checkpoint_path is not None:
                pytorch_result = pytorch_model.test_step(model_inputs)[0]
            backend_result = backend_model.test_step(model_inputs)[0]

            task_processor.visualize(
                image=batch_img,
                model=backend_model,
                result=backend_result,
                output_file=output_file,
                window_name=backend.value,
                show_result=False)
            backend_result_img = mmcv.imread(output_file)

            if checkpoint_path is not None:
                task_processor.visualize(
                    image=batch_img,
                    model=pytorch_model,
                    result=pytorch_result,
                    output_file=output_file,
                    window_name=backend.value,
                    show_result=False)
                pytorch_result_img = mmcv.imread(output_file)

                result = np.concatenate(
                    (backend_result_img, pytorch_result_img), axis=1)
                mmcv.imwrite(result, output_file)


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

        if args.batch < 2:
            batch_visualize_model(model_cfg_path, deploy_cfg_path, single_img,
                                  args.device, deploy_model_path, None,
                                  backend, output_file)
        else:
            batch_visualize_model(model_cfg_path, deploy_cfg_path, single_img,
                                  args.device, deploy_model_path,
                                  checkpoint_path, backend, output_file)


if __name__ == '__main__':
    main()

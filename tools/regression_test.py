# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from pathlib import Path

import yaml
import torch
# from mmdeploy.apis import torch2onnx

from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Process Regression Test')
    parser.add_argument('--deploy-yml', help='regression test yaml path',
                        default='../configs/mmdet/mmdet_regression_test.yaml')
    parser.add_argument('--test-type', help='`test type', default="precision")
    parser.add_argument('--backend', help='test specific backend(s)',
                        default="all")
    parser.add_argument('--work-dir', help='the dir to save logs and models',
                        default='../../mmdeploy_regression_working_dir')
    parser.add_argument('--device-id', help='`the CUDA device id', default=0)
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def get_model_metafile_info(global_info, model_info, logger):
    """Get model metafile information.

    Args:
        global_info (dict): global info from deploy yaml.
        model_info (dict):  model info from deploy yaml.
        logger (logging.Logger): logger.

    Returns:
        Dict: Meta infos of each model config
    """

    # get info from global_info and model_info
    checkpoint_dir = global_info.get('checkpoint_dir', None)
    assert checkpoint_dir is not None

    codebase_dir = global_info.get('codebase_dir', None)
    assert codebase_dir is not None

    codebase_name = global_info.get('codebase_name', None)
    assert codebase_name is not None

    model_config_files = model_info.get('model_configs', [])
    assert len(model_config_files) > 0

    # make checkpoint save directory
    checkpoint_save_dir = Path(checkpoint_dir).joinpath(codebase_name,
                                                        model_info.get('name'))
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    # get model metafile info
    metafile_path = Path(codebase_dir).joinpath(model_info.get('metafile'))
    with open(metafile_path) as f:
        metafile_info = yaml.load(f, Loader=yaml.FullLoader)

    model_meta_info = dict()
    for meta_model in metafile_info.get('Models'):
        if str(meta_model.get('Name')) + ".py" not in model_config_files:
            # skip if the model not in model_config_files
            continue

        # get meta info
        model_meta_info.update({meta_model.get('Name'): meta_model})

        # get weight url
        weights_url = meta_model.get('Weights')
        weights_name = str(weights_url).split('/')[-1]
        weights_save_path = checkpoint_save_dir.joinpath(weights_name)
        if weights_save_path.exists() and \
                not global_info.get('checkpoint_force_download', False):
            continue

        # Download weight
        logger.info(f'Downloading {weights_url} to {weights_save_path}')
        torch.hub.download_url_to_file(weights_url, str(weights_save_path), progress=True)

        # check weather the weight download successful
        if not weights_save_path.exists():
            raise FileExistsError(f'Weight {weights_name} download fail')

    logger.info(f'All models had been downloaded successful.')
    return model_meta_info, checkpoint_save_dir, codebase_dir


def get_pytorch_result(meta_info, model_config_name, logger):
    """Get metric from metafile info of the model

    Args:
        meta_info (dict): metafile info from model's metafile.yml.
        model_config_name (str):  model config name for getting meta info
        logger (logging.Logger): logger.

    Returns:
        Dict: metric info of the model
    """

    if model_config_name not in meta_info:
        return {}

    model_info = meta_info.get(model_config_name, None)
    metric = model_info.get('Results', None)

    logger.info(f'Got {model_config_name} metric: {metric}')
    return metric


def get_onnxruntime_result(backends_info, model_cfg_path, deploy_config_dir, checkpoint_path, work_dir, device, logger):
    """Convert model to onnx and then get metric.

    Args:
        backends_info (dict):  backend info of test yaml.
        model_cfg_path (Path): model config file path.
        deploy_config_dir (str): deploy config directory.
        checkpoint_path (Path): checkpoints path.
        work_dir (Path): A working directory.
        device (str): A string specifying device, defaults to 'cuda:0'.
        logger (logging.Logger): logger.

    Returns:
        Dict: metric info of the model
    """

    # convert
    backends_info = backends_info.get('onnxruntime', [])
    if len(backends_info) <= 0:
        return {}

    deploy_cfg_path_list = backends_info.get('deploy_config')
    for infer_type, deploy_cfg_info in deploy_cfg_path_list.items():
        for fp_size, deploy_cfg in deploy_cfg_info.items():
            img = None
            deploy_cfg_path = Path(deploy_config_dir).joinpath(deploy_cfg)
            logger.info(f'torch2onnx: \n\tmodel_cfg: {model_cfg_path} '
                        f'\n\tdeploy_cfg: {deploy_cfg_path}')
            try:
                torch2onnx(
                    img,
                    work_dir,
                    Path(checkpoint_path).with_suffix('.onnx'),
                    deploy_cfg=deploy_cfg_path,
                    model_cfg=model_cfg_path,
                    model_checkpoint=checkpoint_path,
                    device=device)
                logger.info('torch2onnx success.')
            except Exception as e:
                logger.error(e)
                logger.error('torch2onnx failed.')

    return False


def get_tensorrt_result(global_info, model_info, logger):
    return False


def get_openvino_result(global_info, model_info, logger):
    return False


def get_ncnn_result(global_info, model_info, logger):
    return False


def get_pplnn_result(global_info, model_info, logger):
    return False


def get_sdk_result(global_info, model_info, logger):
    return False


def save_report():
    pass


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    logger.info('Processing regression test.')

    deploy_yaml_list = str(args.deploy_yml).replace(' ', '').split(',')
    assert len(deploy_yaml_list) > 0

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for deploy_yaml in deploy_yaml_list:

        if not Path(deploy_yaml).exists():
            raise FileNotFoundError(f'deploy_yaml {deploy_yaml} not found, '
                                    'please check !')

        with open(deploy_yaml) as f:
            yaml_info = yaml.load(f, Loader=yaml.FullLoader)

        global_info = yaml_info.get('globals')
        models_info = yaml_info.get('models')

        for models in models_info:
            if 'model_configs' not in models:
                continue

            model_metafile_info, checkpoint_save_dir, codebase_dir = \
                get_model_metafile_info(global_info, models, logger)
            for model_config in model_metafile_info:
                logger.info(f'Processing regression test for {model_config}.py...')

                # get model config path
                model_cfg_path = Path(codebase_dir). \
                    joinpath(models.get("codebase_model_config_dir", ""), model_config).with_suffix('.py')
                assert model_cfg_path.exists()

                # get checkpoint path
                checkpoint_name = Path(model_metafile_info.get(model_config).get('Weights')).name
                checkpoint_path = Path(checkpoint_save_dir).joinpath(checkpoint_name)
                assert checkpoint_path.exists()

                # get deploy config directory
                deploy_config_dir = models.get('deploy_config_dir', '')
                assert deploy_config_dir != ''

                # get backends info
                backends_info = models.get('backends', None)
                if backends_info is None:
                    continue

                pytorch_result = get_pytorch_result(model_metafile_info, model_config, logger)
                onnxruntime_result = get_onnxruntime_result(backends_info, model_cfg_path, deploy_config_dir,
                                                            checkpoint_path, work_dir, args.device_id, logger)
                tensorrt_result = get_tensorrt_result(global_info, models, logger)
                openvino_result = get_openvino_result(global_info, models, logger)
                ncnn_result = get_ncnn_result(global_info, models, logger)
                pplnn_result = get_pplnn_result(global_info, models, logger)
                sdk_result = get_sdk_result(global_info, models, logger)

            save_report()


if __name__ == '__main__':
    main()

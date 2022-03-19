# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from mmdeploy.utils import get_root_logger


# from mmdeploy.apis import torch2onnx


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


def update_report(report_dict,
                  model_name,
                  model_config,
                  model_checkpoint_name,
                  dataset,
                  backend_name,
                  deploy_config,
                  static_or_dynamic,
                  conversion_result,
                  fps,
                  metric_info,
                  test_pass,
                  ):
    """Update report information

    Args:
        report_dict (dict): Report info dict.
        model_name (str): Model name.
        model_config (str): Model config name.
        model_checkpoint_name (str): Model checkpoint name.
        dataset (str): Dataset name.
        backend_name (str): Backend name.
        deploy_config (str): Deploy config name.
        static_or_dynamic (str): Static or dynamic.
        conversion_result (str): Conversion result: Successful or Fail.
        fps (str): Inference speed (ms/im).
        metric_info (list): Metric info list of the ${modelName}.yml.
        test_pass (str): Test result: Pass or Fail.
    """
    report_dict.get('model_name').append(model_name)
    report_dict.get('model_config').append(model_config)
    report_dict.get('model_checkpoint_name').append(model_checkpoint_name)
    report_dict.get('dataset').append(dataset)
    report_dict.get('backend_name').append(backend_name)
    report_dict.get('deploy_config').append(deploy_config)
    report_dict.get('static_or_dynamic').append(static_or_dynamic)
    report_dict.get('conversion_result').append(conversion_result)
    report_dict.get('fps').append(fps)

    for metric in metric_info:
        for metric_name, metric_value in metric.items():
            metric_name = str(metric_name).replace(' ', '_')
            report_dict.get(metric_name).append(metric_value)

    report_dict.get('test_pass').append(test_pass)


def get_pytorch_result(model_name, meta_info, checkpoint_name, model_config_name, metric_all_list, report_dict, logger):
    """Get metric from metafile info of the model

    Args:
        model_name (str): Name of model.
        meta_info (dict): Metafile info from model's metafile.yml.
        checkpoint_name (str):  Name of checkpoint.
        metric_all_list (list): All metric name.
        model_config_name (str):  Model config name for getting meta info
        report_dict (dict): Report info dict.
        logger (logging.Logger): Logger.

    Returns:
        List: metric info of the model
    """

    if model_config_name not in meta_info:
        return {}

    model_info = meta_info.get(model_config_name, None)

    # get metric
    metric_info = model_info.get('Results', None)
    metric_list = []
    for metric in metric_info:
        metric_list.append(metric.get('Metrics'))

    # update useless metric
    metric_useless = set(metric_all_list) - \
                     set([str(list(metric.keys())[0]).replace(' ', '_') for metric in metric_list])
    for metric in metric_useless:
        metric_list.append({metric: '-'})

    # get pytorch fps value
    fps_info = model_info.get('Metadata').get('inference time (ms/im)')
    if fps_info is None:
        fps = "-"
    elif isinstance(fps_info, list):
        fps = fps_info[0].get('value')
    else:
        fps = fps_info.get('value')

    # update report
    dataset_type = ""
    for metric in metric_info:
        dataset_type += f"{metric.get('Dataset')},"

    update_report(
        report_dict=report_dict,
        model_name=model_name,
        model_config=model_config_name,
        model_checkpoint_name=checkpoint_name,
        dataset=dataset_type,
        backend_name='Pytorch',
        deploy_config='-',
        static_or_dynamic='-',
        conversion_result='-',
        fps=fps,
        metric_info=metric_list,
        test_pass='-'
    )

    logger.info(f'Got {model_config_name} metric: {metric_list}')
    return metric_list


def get_onnxruntime_result(backends_info, model_cfg_path, deploy_config_dir, checkpoint_path,
                           work_dir, device, metric_all_list, report_dict, logger):
    """Convert model to onnx and then get metric.

    Args:
        backends_info (dict):  Backend info of test yaml.
        model_cfg_path (Path): Model config file path.
        deploy_config_dir (str): Deploy config directory.
        checkpoint_path (Path): Checkpoints path.
        work_dir (Path): A working directory.
        device (str): A string specifying device, defaults to 'cuda:0'.
        metric_all_list (list): All metric name.
        report_dict (dict): Report info dict.
        logger (logging.Logger): Logger.

    Returns:
        Dict: metric info of the model
    """

    backends_info = backends_info.get('onnxruntime', [])
    if len(backends_info) <= 0:
        return {}

    metric_name_list = backends_info.get('metric', [])
    assert len(metric_name_list) > 0
    test_pass = '-'

    deploy_cfg_path_list = backends_info.get('deploy_config')
    for infer_type, deploy_cfg_info in deploy_cfg_path_list.items():
        for fp_size, deploy_cfg in deploy_cfg_info.items():

            # convert
            fps = ''
            convert_result = True
            onnxruntime_path = Path(checkpoint_path).with_suffix('.onnx')
            img = None
            metric_list = []
            deploy_cfg_path = Path(deploy_config_dir).joinpath(deploy_cfg)
            logger.info(f'torch2onnx: \n\tmodel_cfg: {model_cfg_path} '
                        f'\n\tdeploy_cfg: {deploy_cfg_path}')

            try:
                torch2onnx(
                    img,
                    work_dir,
                    onnxruntime_path,
                    deploy_cfg=deploy_cfg_path,
                    model_cfg=model_cfg_path,
                    model_checkpoint=checkpoint_path,
                    device=device)
                logger.info('torch2onnx success.')

            except Exception as e:
                logger.error(e)
                logger.error('torch2onnx failed.')
                convert_result = False

            finally:
                if convert_result:
                    for metric in metric_list:
                        # test the model
                        # metric_dict.update({metric: 0})
                        # fps = 0
                        # metric_tolerance
                        pass
                else:
                    for metric in metric_name_list:
                        metric_list.append({metric: '-'})

                # update useless metric
                metric_useless = set(metric_all_list) - set(metric_name_list)
                for metric in metric_useless:
                    metric_list.append({metric: '-'})

                update_report(
                    report_dict=report_dict,
                    model_name=model_cfg_path.parent.name,
                    model_config=str(model_cfg_path),
                    model_checkpoint_name=str(checkpoint_path),
                    dataset='',
                    backend_name='onnxruntime',
                    deploy_config=str(deploy_cfg_path),
                    static_or_dynamic=infer_type,
                    conversion_result=str(convert_result),
                    fps=fps,
                    metric_info=metric_list,
                    test_pass=test_pass
                )


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


def save_report(report_info, report_save_path, logger):
    """Convert model to onnx and then get metric.

    Args:
        report_info (dict):  Report info dict.
        report_save_path (Path): Report save path.
        logger (logging.Logger): Logger.
    """
    logger.info(f'Save regression test report '
                f'to {report_save_path}, pls wait...')

    df = pd.DataFrame(report_info)
    df.to_excel(report_save_path)

    logger.info(f'Saved regression test report '
                f'to {report_save_path}.')


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

        report_save_path = work_dir.joinpath(Path(deploy_yaml).stem + '_report.xlsx')
        report_dict = {
            'model_name': [],
            'model_config': [],
            'model_checkpoint_name': [],
            'dataset': [],
            'backend_name': [],
            'deploy_config': [],
            'static_or_dynamic': [],
            'conversion_result': [],
            'fps': []
        }

        global_info = yaml_info.get('globals')

        metric_all_list = []
        for metric_name in global_info.get('metric_tolerance', {}):
            report_dict.update({metric_name: []})
            metric_all_list.append(metric_name)

        report_dict.update({'test_pass': []})

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

                pytorch_result = get_pytorch_result(models.get('name'), model_metafile_info, checkpoint_name,
                                                    model_config, metric_all_list, report_dict, logger)
                get_onnxruntime_result(backends_info, model_cfg_path, deploy_config_dir,
                                       checkpoint_path, work_dir, args.device_id, metric_all_list,
                                       report_dict, logger)
                tensorrt_result = get_tensorrt_result(global_info, models, logger)
                openvino_result = get_openvino_result(global_info, models, logger)
                ncnn_result = get_ncnn_result(global_info, models, logger)
                pplnn_result = get_pplnn_result(global_info, models, logger)
                sdk_result = get_sdk_result(global_info, models, logger)

        save_report(report_dict, report_save_path, logger)


if __name__ == '__main__':
    main()

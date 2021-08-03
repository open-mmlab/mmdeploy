import warnings
from typing import Any, Union

import mmcv
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from mmdeploy.apis.utils import assert_module_exist


def prepare_data_loader(codebase: str, model_cfg: Union[str, mmcv.Config]):
    # load model_cfg if necessary
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)

    if codebase == 'mmcls':
        from mmcls.datasets import (build_dataloader, build_dataset)
        assert_module_exist(codebase)
        # build dataset and dataloader
        dataset = build_dataset(model_cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=model_cfg.data.samples_per_gpu,
            workers_per_gpu=model_cfg.data.workers_per_gpu,
            shuffle=False,
            round_up=False)

    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        from mmdet.datasets import (build_dataloader, build_dataset,
                                    replace_ImageToTensor)
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(model_cfg.data.test, dict):
            model_cfg.data.test.test_mode = True
            samples_per_gpu = model_cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                model_cfg.data.test.pipeline = replace_ImageToTensor(
                    model_cfg.data.test.pipeline)
        elif isinstance(model_cfg.data.test, list):
            for ds_cfg in model_cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max([
                ds_cfg.pop('samples_per_gpu', 1)
                for ds_cfg in model_cfg.data.test
            ])
            if samples_per_gpu > 1:
                for ds_cfg in model_cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        # build the dataloader
        dataset = build_dataset(model_cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=model_cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')

    return dataset, data_loader


def single_gpu_test(codebase: str,
                    model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Any = None,
                    show_score_thr: float = 0.3):
    if codebase == 'mmcls':
        assert_module_exist(codebase)
        from mmcls.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir,
                                  show_score_thr)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')
    return outputs


def post_process_outputs(outputs,
                         dataset,
                         model_cfg: mmcv.Config,
                         codebase: str,
                         metrics: str = None,
                         out: str = None,
                         metric_options: dict = None,
                         format_only: bool = False):
    if codebase == 'mmcls':
        if metrics:
            results = dataset.evaluate(outputs, metrics, metric_options)
            for k, v in results.items():
                print(f'\n{k} : {v:.2f}')
        else:
            warnings.warn('Evaluation metrics are not specified.')
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [dataset.CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(results, out)

    elif codebase == 'mmdet':
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')

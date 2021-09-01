import warnings
from typing import Any

import mmcv
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from mmdeploy.utils import Codebase


def single_gpu_test(codebase: str,
                    model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Any = None,
                    show_score_thr: float = 0.3):
    if codebase == Codebase.MMCLS:
        from mmcls.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMDET:
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir,
                                  show_score_thr)
    elif codebase == Codebase.MMSEG:
        from mmseg.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMOCR:
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMEDIT:
        from mmedit.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')
    return outputs


def post_process_outputs(outputs,
                         dataset,
                         model_cfg: mmcv.Config,
                         codebase: str,
                         metrics: str = None,
                         out: str = None,
                         metric_options: dict = None,
                         format_only: bool = False):
    if codebase == Codebase.MMCLS:
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

    elif codebase == Codebase.MMDET:
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

    elif codebase == Codebase.MMSEG:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            dataset.evaluate(outputs, metrics, **kwargs)

    elif codebase == Codebase.MMOCR:
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

    elif codebase == Codebase.MMEDIT:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        # The Dataset doesn't need metrics
        print('')
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')

# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time

import yaml

assert os.path.exists('checkpoints')
assert os.path.exists('images')
if os.path.isfile('report.txt'):
    with open('report.txt', 'a') as f:
        sys.stdout = f
        for i in range(50):
            print('')
        f.close()
yaml_file = 'tools/benchmark_test.yml'
with open(yaml_file) as f:
    benchmark_test_info = yaml.load(f, Loader=yaml.FullLoader)
for model in benchmark_test_info['models']:
    for model_info in model['model_info']:
        for deploy_cfg in model['deploy_cfg']:
            if 'int8' in deploy_cfg:
                assert os.path.exists('data/coco')
            model_cfg = model_info['model_cfg']
            checkpoint = model_info['checkpoint']
            shape = model_info['shape']
            img_path = f'images/{shape}.jpg'
            backend_model = model['backend_model']
            device = model['device']
            img_folder = 'images/data'
            assert os.path.exists(img_path)
            assert os.path.exists(img_folder)
            convert_cmd = (f'python tools/deploy.py {deploy_cfg} ' +
                           f'{model_cfg} {checkpoint} {img_path} ' +
                           f' --work-dir tools/ --device {device}')
            os.system(f'rm -rf tools/{backend_model}')
            with open('report.txt', 'a') as f:
                sys.stdout = f
                print(convert_cmd)
                os.system('nvidia-smi >> report.txt')
                f.close()
            os.system(convert_cmd)
            profile_cmd = (f'python tools/profiler.py {deploy_cfg} ' +
                           f'{model_cfg} {img_folder} --model tools/' +
                           f'{backend_model} --device {device} --shape ' +
                           f'{shape} >> report.txt')
            with open('report.txt', 'a') as f:
                sys.stdout = f
                print(profile_cmd)
                time.sleep(3)
                os.system(profile_cmd)
                f.close()

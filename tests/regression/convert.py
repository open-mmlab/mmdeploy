# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import numpy as np
import yaml
from easydict import EasyDict as edict
from tomark import Tomark

from mmdeploy.utils import get_backend, get_codebase, get_task_type


def parse_args():
    parser = argparse.ArgumentParser(description='From yaml export markdown')
    parser.add_argument('yaml_cfg', help='yml config path')
    parser.add_argument('output_md_file', help='Output markdown file path')
    args = parser.parse_args()
    return args


def generate_inference_dict():
    args = parse_args()
    inference_dict = {}
    name_list = []
    metafile_list = []
    model_configs_list = []
    piplines_list = []

    with open(args.yaml_cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
        for k, v in config.items():
            if k == 'models':
                for model in v:
                    name = model['name']
                    metafile = model['metafile']
                    model_configs = model['model_configs']
                    piplines = model['pipelines']
                    name_list.append(name)
                    metafile_list.append(metafile)
                    model_configs_list.append(model_configs)
                    piplines_list.append(piplines)
                    list = []
                    for content in piplines:

                        deploy_config = content['deploy_config']
                        (file, ext) = osp.splitext(deploy_config)
                        platform = osp.basename(file)
                        inference_platform = platform.split('_')[1]
                        list.append(inference_platform)
                        inference_dict[name] = list
    return inference_dict, name_list, model_configs_list, piplines_list


def parse_deploy_config():

    _, _, _, piplines_list = generate_inference_dict()
    codebase_list = []
    backend_list = []
    get_task_type_list = []
    for piplines in piplines_list:
        for i in piplines:
            codebase = get_codebase(i['deploy_config'])
            backend = get_backend(i['deploy_config'])
            task_type = get_task_type(i['deploy_config'])

            codebase_list.append(codebase)
            backend_list.append(backend)
            get_task_type_list.append(task_type)
    return codebase_list, backend_list, get_task_type_list


def website_list():
    model_website_list = []
    _, _, model_configs_list, _ = generate_inference_dict()
    _, _, get_task_type_list = parse_deploy_config()

    task_type = str(get_task_type_list[0])
    task_type = task_type.split('.', 1)[1]
    if task_type == 'OBJECT_DETECTION':
        task_type = task_type.split('_', 1)[1]
        task_type = task_type.lower()
    else:
        task_type = task_type.lower()
    model_website_task = f'mm{task_type}'
    for i in model_configs_list:
        name = str(i[0])
        model_name = osp.split(name)[0]
        model_website_list.append(model_name)
    return model_website_list, model_website_task


def main():
    args = parse_args()
    inference_list = []

    inference_dict, name_list, model_configs_list, _ = generate_inference_dict(
    )
    _, backend_list, get_task_type_list = parse_deploy_config()
    model_website_list, model_website_task = website_list()
    platform1 = []
    i = 0
    for a in backend_list:
        a = str(backend_list[i])
        y = a.split('.', 1)[1]
        y = y.lower()
        platform1.append(y)
        i += 1
    platform1 = np.unique(platform1)
    print(platform1)

    model_configs_list
    task_type = str(get_task_type_list[0])
    task_type = task_type.split('.', 1)[1]

    x = 0
    for name in name_list:

        dict = {}
        website_name = model_website_list[x]
        x += 1
        front = 'https://github.com/open-mmlab/'
        if model_website_task == 'mmdetection':

            url = f'{front}{model_website_task}/tree/3.x/'
            url_name = f'{url}{website_name}'
            dict['Model'] = f'[{name}]({url_name})'
            task_type = task_type.title()
            dict['Task'] = task_type
            inference_list.append(dict)
        else:
            url = f'{front}{model_website_task}/tree/1.x/'
            url_name = f'{url}{website_name}'
            dict['Model'] = f'[{name}]({url_name})'
            task_type = task_type.title()
            dict['Task'] = task_type
            inference_list.append(dict)

    target_platform = []
    for list in inference_dict:
        a = inference_dict[list]
        target_platform.append(a)

    for j in range(len(inference_list)):

        dict = inference_list[j]
        for i in platform1:
            if i in target_platform[j]:
                dict[i] = 'Y'
            else:
                dict[i] = 'N'

    markdown = Tomark.table(inference_list)
    print(markdown)
    path = args.output_md_file
    (file, ext) = osp.splitext(path)

    with open(f'{file}.md', 'w') as f:
        f.write(markdown)
        f.close()


if __name__ == '__main__':
    main()

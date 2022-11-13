import argparse
import os
import os.path as osp

import mmcv
import yaml
from easydict import EasyDict as edict
from markdown.extensions.tables import TableExtension
from mdutils.mdutils import MdUtils
from tomark import Tomark

from mmdeploy.utils import (get_backend, get_codebase, get_task_type,
                            load_config)

import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='From yaml export markdown')
    parser.add_argument('yaml_cfg',help='yml config path')
    parser.add_argument('output_md_file',help='Output markdown file path')
    args = parser.parse_args()
    return args

def generate_inference_dict():
    args = parse_args()
    inference_dict = {}
    name_list = []
    metafile_list = []
    model_configs_list = []
    piplines_list = []

    ##inference_dict
    with open(args.yaml_cfg,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
        for k, v in config.items():
            if k == "models":
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
                        (file,ext) = osp.splitext(deploy_config)
                        platform = osp.basename(file)
                        inference_platform = platform.split("_")[1]
                        list.append(inference_platform)
                        inference_dict[name] =  list
    return inference_dict,name_list,metafile_list,model_configs_list,piplines_list

def parse_deploy_config():

    inference_dict,name_list,metafile_list,model_configs_list,piplines_list = generate_inference_dict()
    codebase_list = []
    backend_list = []
    get_task_type_list = [] 
    for piplines in piplines_list:
        for i in piplines:
            #print(i['deploy_config'])   
            codebase = get_codebase(i['deploy_config'])
            backend = get_backend(i['deploy_config'])
            task_type = get_task_type(i['deploy_config'])

            codebase_list.append(codebase)
            backend_list.append(backend)
            get_task_type_list.append(task_type)
    return codebase_list,backend_list,get_task_type_list

#backend = get_backend(model_configs_list[0][0])

def main():
    args = parse_args()
    inference_list = []
    
    inference_dict,name_list,metafile_list,model_configs_list,piplines_list = generate_inference_dict()
    codebase_list,backend_list,get_task_type_list = parse_deploy_config()
    
    platform1 = []
    i = 0
    for a in backend_list:
        a = str(backend_list[i])
        y = a.split('.',1)[1]
        y = y.lower()
        platform1.append(y)
        i+=1
    platform1 = np.unique(platform1)
    print(platform1)

    #print(get_task_type_list)
    #print(backend_list,codebase_list[0],get_task_type_list[0])
    
    #platform = ['torchscript', 'onnxruntime', 'tensorrt-fp16', 'ncnn', 'pplnn', 'openvino','tensorrt']

    model_configs_list
    task_type = str(get_task_type_list[0])
    task_type = task_type.split('.',1)[1]
    
    if task_type == 'CLASSIFICATION':
        web = 'mmclassification'
    elif task_type == 'SEGMENTATION':
        web = 'mmsegmentation'
    else :
        web = 'mmdetection'

    for name in name_list:
        
        dict = {}
        lower_name = name.lower()
        if lower_name == 'semantic fpn':
            lower_name = 'sem_fpn'
            url = f'https://github.com/open-mmlab/{web}/tree/1.x/configs/'
            url_name = f'{url}{lower_name}'
            dict['Model'] = f'[{name}]({url_name})'
            dict['Task'] = task_type
            inference_list.append(dict)
        
        elif web == 'mmdetection':
            url = f'https://github.com/open-mmlab/{web}/tree/dev-3.x/configs/'
            url_name = f'{url}{lower_name}'
            dict['Model'] = f'[{name}]({url})'
            dict['Task'] = task_type
            inference_list.append(dict)
        else:
            url = f'https://github.com/open-mmlab/{web}/tree/1.x/configs/'
            url_name = f'{url}{lower_name}'
            dict['Model'] = f'[{name}]({url_name})'
            dict['Task'] = task_type
            inference_list.append(dict)
    

    target_platform=[]
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
    (file,ext) = osp.splitext(path)
    f = open(f'{file}.md','w')
    f.write(markdown)
    f.close()



if __name__ == '__main__':
    main()


            

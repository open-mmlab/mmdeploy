import argparse
import yaml
from easydict import EasyDict as edict
import os
import os.path as osp
from markdown.extensions.tables import TableExtension
import mmcv

from tomark import Tomark
from mdutils.mdutils import MdUtils

from mmdeploy.utils import get_codebase,get_backend,get_task_type


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

def parse_deploy_config(deploy_cfg):

    inference_dict,name_list,metafile_list,model_configs_list,piplines_list = generate_inference_dict()
    codebase_list = []
    backend_list = []
    get_task_type_list = [] 
    for model_config in model_configs_list:   
        codebase = get_codebase(model_config[0])
        backend = get_backend(model_config[0])
        task_type = get_task_type(model_config[0])
        codebase_list.append(codebase)
        backend_list.append(backend)
        get_task_type_list.append(task_type)
    return codebase_list,backend_list,get_task_type_list

inference_dict,name_list,metafile_list,model_configs_list,piplines_list = generate_inference_dict()
backend = get_backend(model_configs_list[0][0])

def main():
    args = parse_args()
    inference_list = []
    inference_dict,name_list,metafile_list,model_configs_list,piplines_list = generate_inference_dict()

    platform = ['torchscript', 'onnxruntime', 'tensorrt-fp16', 'ncnn', 'pplnn', 'openvino','tensorrt']

    model_configs_list
    i = 0
    for name in name_list:
        dict = {}
        lower_name = name.lower()
        url = 'https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/'
        url_name = f'{url}{lower_name}'
        dict['Model'] = f'[{name}]({url_name})'
        dict['Task'] = 'Segmenter'
        inference_list.append(dict)
        i += 1

    target_platform=[]
    for list in inference_dict:
        a = inference_dict[list]
        target_platform.append(a)

    for j in range(len(inference_list)):
        
        dict = inference_list[j]    
        for i in platform:
            if i in target_platform[j]:
                dict[i] = 'Y'
            else:
                dict[i] = 'N'
            
    markdown = Tomark.table(inference_list)
    print(markdown)
    path = args.output_md_file
    (file,ext) = osp.splitext(path)
    f = open(f'{path}.md','w')
    f.write(markdown)
    f.close()


if __name__ == '__main__':
    main()


                                      

                    
                    









            

import yaml
from easydict import EasyDict as edict
import os
import os.path as osp
from markdown.extensions.tables import TableExtension
import markdown as M

import pandas as pd
from tomark import Tomark
from mdutils.mdutils import MdUtils

inference_dict = {}
name_list = []
inference_list = []
metafile_list = []
model_configs_list = []
piplines_list = []

##name_list
with open('mmseg.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    for k, v in config.items():
        ##   print(k)
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


##inference_dict
with open('mmseg.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)
    for k, v in config.items():
        ##   print(k)
        if k == "models":
            for model in v:
                name = model['name']
                metafile = model['metafile']
                model_configs = model['model_configs']
                piplines = model['pipelines']
                
                #print(name+':')
                #print(model_configs)
                #inference_dict + name
                list = [] 
                for content in piplines:
                    
                    deploy_config = content['deploy_config']
                    ##print(deploy_config)
                    (file,ext) = osp.splitext(deploy_config)
                    platform = osp.basename(file)
                    #print(platform)
                    inference_platform = platform.split("_")[1]
                    list.append(inference_platform)

                    #html = markdown.markdown(inference_platform,extensions=[TableExtension(use_align_attribute=True)])
                    #dict[name] = inference_platform
                    #list.append(inference_platform)
                    inference_dict[name] =  list
                    #inference_list.append(dict)

#print(inference_dict)

#print(name_list)
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
    a=inference_dict[list]
    #print(a)
    target_platform.append(a)

#print(target_platform)
## platform2model
for j in range(len(inference_list)):
    
    dict = inference_list[j]
        #dict[i] = ''
    
    for i in platform:
        if i in target_platform[j]:
            dict[i] = 'Y'

        else:
            dict[i] = 'N'
        

    
        '''
        for list in inference_dict:

            target_platform = inference_dict[list]
            if i in target_platform:
                dict[i] = True
            else:
                dict[i] = False '''

#print(inference_list)

#inference_dict
markdown = Tomark.table(inference_list)
print(markdown)
f = open('result.md','w')
f.write(markdown)
f.close()







                                      

                    
                    









            

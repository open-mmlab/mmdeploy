# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import re

import requests

URL = 'http://139.196.52.65:8089/webhook/event'
ROOT_DIR = '/data2/regression_log/convert_log/20221128/202211282213'
FILE_DIR = 'http://10.1.52.36:8989/convert_log/20221128/202211282213'
DOCUMENT_ID = 'https://aicarrier.feishu.cn/docx/ZJA3dlpsjosXJRxXsF3cqUXqnue'


def GetCodebases(root_dir):
    files = os.listdir(root_dir)
    tmp = []
    for file in files:
        if re.search('mm.*?', file) and not re.search('mm.*?\..*?', file):
            tmp.append(file)
    return tmp


def GetTorchs(root_dir, codebase):
    files = os.listdir(f'{root_dir}/{codebase}')
    tmp = []
    for file in files:
        if re.search('torch.*?', file):
            tmp.append(file)
    return tmp


def GetLogDir(root_dir, codebase, torch):
    return f'{root_dir}/{codebase}/{torch}/{codebase}_report.txt'


def GetResultList(log_dir):
    # index=0, value=Model
    # index=1, value=Model Config
    # index=2, value=Task
    # index=3, value=Checkpoint
    # index=4, value=Dataset
    # index=5, value=Backend
    # index=6, value=Deploy Config
    # index=7, value=Static or Dynamic
    # index=8, value=Precision Type
    # index=9, value=Conversion Result
    # index=10, value=box AP
    # index=11, value=mask AP
    # index=12, value=PQ
    # index=13, value=Test Pass

    with open(log_dir, 'r') as f:
        data = f.readlines()
    result_list = []
    for i in range(1, len(data)):
        result_list.append(data[i].split(','))
    return result_list


def GetFalseList(result_list):
    false_list = []
    for result in result_list:
        if 'False' in result:
            false_list.append(result)

    return false_list


def CreateResultWord(args):
    # return url
    # request lark_bot to get url
    pass


def CreateDocument(title):
    payload = json.dumps({
        'header': {
            'event_type': 'document'
        },
        'event': {
            'ops': 'create_document',
            'title': title
        }
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request('POST', URL, headers=headers, data=payload)
    return response.json()['ret']


def CreateBlock(document_id, block_id, content):
    payload = json.dumps({
        'header': {
            'event_type': 'document'
        },
        'event': {
            'ops': 'create_block',
            'document_id': document_id,
            'block_id': block_id,
            'content': content
        }
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request('POST', URL, headers=headers, data=payload)


def main_ops(root_dir):
    error_count = 0
    report_list = []
    codebases = GetCodebases(root_dir)
    for codebase in codebases:
        torchs = GetTorchs(root_dir, codebase)
        for torch in torchs:
            log_dir = GetLogDir(root_dir, codebase, torch)
            result_list = GetResultList(log_dir)
            false_list = GetFalseList(result_list)
            content = ''
            if len(false_list) > 0:
                for a_list in false_list:
                    error_count += 1
                    model = a_list[0].lower().replace('-', '')
                    backend = a_list[5]

                    detail_log_path = f'{FILE_DIR}/{codebase}/{torch}/{codebase}/{model}/{backend}/{a_list[7]}/{a_list[8]}'
                    if a_list[3] != 'x' and a_list[3] != 'x.param':
                        checkpoint = re.findall(r'.*/(.*)\..*', a_list[3])

                        detail_log_path_full = f'{detail_log_path}/{checkpoint}'
                    else:
                        detail_log_path_full = detail_log_path
                    tmp = f"""
model: {a_list[0]}
model config: {a_list[1]}
backend: {a_list[5]}
log_dir: {detail_log_path_full}
            """
                    content = content + tmp
                if content != '':
                    date = re.findall(r'.*/(.*)', root_dir)[0]
                    title = f'{date}-{codebase}-{torch}'
                    # document_id = CreateDocument(title)
                    CreateBlock(
                        document_id=DOCUMENT_ID,
                        block_id=DOCUMENT_ID,
                        content=content)
                    report_list.append(f'{content}')
    for i in report_list:
        print(i)
    # print(f'总错误条数: {error_count}')


main_ops(ROOT_DIR)

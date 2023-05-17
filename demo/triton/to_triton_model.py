# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
from enum import Enum, unique
import shutil
from glob import glob

BASEDIR = os.path.dirname(__file__)


@unique
class Template(str, Enum):
    ImageClassification = 'image-classification/serving'
    InstanceSegmentation = 'instance-segmentation/serving'
    KeypointDetection = 'keypoint-detection/serving'
    ObjectDetection = 'object-detection/serving'
    OrientedObjectDetection = 'oriented-object-detection/serving'
    SemanticSegmentation1 = 'semantic-segmentation/serving/mask'
    SemanticSegmentation2 = 'semantic-segmentation/serving/score'
    TextRecognition = 'text-recognition/serving'
    TextDetection = 'text-detection/serving'


def copy_template(src_folder, dst_folder):
    files = glob(osp.join(src_folder, '*'))
    for src in files:
        dst = osp.join(dst_folder, osp.basename(src))
        if osp.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)


class Convert:

    def __init__(self, model_type, model_dir, deploy_cfg, pipeline_cfg, detail_cfg, output_dir):
        self._model_type = model_type
        self._model_dir = model_dir
        self._deploy_cfg = deploy_cfg
        self._pipeline_cfg = pipeline_cfg
        self._detail_cfg = detail_cfg
        self._output_dir = output_dir

    def copy_file(self, file_name, src_folder, dst_folder):
        src_path = osp.join(src_folder, file_name)
        dst_path = osp.join(dst_folder, file_name)
        if osp.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)

    def write_json_file(self, data, file_name, dst_folder):
        dst_path = osp.join(dst_folder, file_name)
        with open(dst_path, 'w') as f:
            json.dump(data, f, indent=4)

    def create_single_model(self):
        output_model_folder = osp.join(self._output_dir, 'model', '1')
        if (self._model_type == Template.TextRecognition):
            self._pipeline_cfg['pipeline']['input'].append('bbox')
            self._pipeline_cfg['pipeline']['tasks'][0]['input'] = ['patch']
            warpbbox = {
                "type": "Task",
                "module": "WarpBbox",
                "input": [
                    "img",
                    "bbox"
                ],
                "output": [
                    "patch"
                ]
            }
            self._pipeline_cfg['pipeline']['tasks'].insert(0, warpbbox)
            self.write_json_file(self._pipeline_cfg,
                                 'pipeline.json', output_model_folder)
        else:
            self.copy_file('pipeline.json', self._model_dir,
                           output_model_folder)

        self.copy_file('deploy.json', self._model_dir, output_model_folder)
        models = self._deploy_cfg['models']
        for model in models:
            net = model['net']
            self.copy_file(net, self._model_dir, output_model_folder)
        for custom in self._deploy_cfg['customs']:
            self.copy_file(custom, self._model_dir, output_model_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
                        help='converted model dir with `--dump-info` flag when convert the model')
    parser.add_argument('output_dir', type=str,
                        help='output dir')
    return parser.parse_args()


def get_model_type(detail_cfg, pipeline_cfg):
    task = detail_cfg['codebase_config']['task']
    output_names = detail_cfg['onnx_config']['output_names']

    if task == 'Classification':
        return Template.ImageClassification
    if task == 'ObjectDetection':
        if 'masks' in output_names:
            return Template.InstanceSegmentation
        else:
            return Template.ObjectDetection
    if task == 'Segmentation':
        with_argmax = pipeline_cfg['pipeline']['tasks'][-1]['params'].get(
            'with_argmax', True)
        if with_argmax:
            return Template.SemanticSegmentation1
        else:
            return Template.SemanticSegmentation2
    if task == 'PoseDetection':
        return Template.KeypointDetection
    if task == 'RotatedDetection':
        return Template.OrientedObjectDetection
    if task == 'TextRecognition':
        return Template.TextRecognition
    if task == 'TextDetection':
        return Template.TextDetection

    assert 0, f'doesn\'t support task {task} with output_names: {output_names}'


if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir

    # check
    assert osp.isdir(model_dir), f'model dir {model_dir} doesn\'t exist'
    info_files = ['deploy.json', 'pipeline.json', 'detail.json']
    for file in info_files:
        path = osp.join(model_dir, file)
        assert osp.exists(path), f'{path} doesn\'t exist in {model_dir}'

    with open(osp.join(model_dir, 'deploy.json')) as f:
        deploy_cfg = json.load(f)
    with open(osp.join(model_dir, 'pipeline.json')) as f:
        pipeline_cfg = json.load(f)
    with open(osp.join(model_dir, 'detail.json')) as f:
        detail_cfg = json.load(f)
        assert 'onnx_config' in detail_cfg, f'currently, only support onnx as middle ir'

    # process
    model_type = get_model_type(detail_cfg, pipeline_cfg)
    convert = Convert(model_type, model_dir, deploy_cfg, pipeline_cfg,
                      detail_cfg, output_dir)

    src_folder = osp.join(BASEDIR, model_type.value)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    copy_template(src_folder, output_dir)
    convert.create_single_model()

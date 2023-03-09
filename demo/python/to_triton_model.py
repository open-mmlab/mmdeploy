# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

import tritonclient.grpc.model_config_pb2 as pb
from google.protobuf import text_format


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--nocopy', type=bool)
    return parser.parse_args()


IMAGE_INPUT = [
    dict(
        name='ori_img',
        dtype=pb.TYPE_UINT8,
        format=pb.ModelInput.FORMAT_NHWC,
        dims=[-1, -1, 3]),
    dict(name='pix_fmt', dtype=pb.TYPE_INT32, dims=[1], optional=True)
]

TASK_OUTPUT = dict(
    Preprocess=[
        dict(name='img', dtype=pb.TYPE_FP32, dims=[3, -1, -1]),
        dict(name='img_metas', dtype=pb.TYPE_STRING, dims=[1])
    ],
    Classifier=[
        dict(name='scores', dtype=pb.TYPE_FP32, dims=[-1, 1]),
        dict(name='label_ids', dtype=pb.TYPE_FP32, dims=[-1, 1])
    ],
    Detector=[dict(name='dets', dtype=pb.TYPE_FP32, dims=[-1, 1])],
    Segmentor=[
        dict(name='mask', dtype=pb.TYPE_INT32, dims=[-1, -1]),
        dict(name='score', dtype=pb.TYPE_FP32, dims=[-1, -1, -1])
    ],
    Restorer=[
        dict(name='output', dtype=pb.TYPE_FP32, dims=[-1, -1, 3])
    ],
    TextDetector=[],
    TextRecognizer=[],
    PoseDetector=[],
    RotatedDetector=[],
    TextOCR=[],
    DetPose=[])


def add_input(model_config, params):
    p = model_config.input.add()
    p.name = params['name']
    p.data_type = params['dtype']
    p.dims.extend(params['dims'])
    if 'format' in params:
        p.format = params['format']
    if 'optional' in params:
        p.optional = params['optional']


def add_output(model_config, params):
    p = model_config.output.add()
    p.name = params['name']
    p.data_type = params['dtype']
    p.dims.extend(params['dims'])


def serialize_model_config(model_config):
    return text_format.MessageToString(
        model_config,
        use_short_repeated_primitives=True,
        use_index_order=True,
        print_unknown_fields=True)


def create_model_config(name, task, backend=None, platform=None):
    model_config = pb.ModelConfig()
    if backend:
        model_config.backend = backend
    if platform:
        model_config.platform = platform
    model_config.name = name
    model_config.max_batch_size = 0

    for input in IMAGE_INPUT:
        add_input(model_config, input)
    for output in TASK_OUTPUT[task]:
        add_output(model_config, output)
    return model_config


def create_preprocess_model():
    pass


def get_onnx_io_names(detail_info):
    onnx_config = detail_info['onnx_config']
    return onnx_config['input_names'], onnx_config['output_names']


def create_inference_model(deploy_info, pipeline_info, detail_info):
    if 'pipeline' in pipeline_info:
        # old-style pipeline specification
        pipeline = pipeline_info['pipeline']['tasks']
    else:
        pipeline = pipeline_info['tasks']

    for task_cfg in pipeline:
        if task_cfg['module'] == 'Net':
            input_names, output_names = get_onnx_io_names(detail_info)


def create_postprocess_model():
    pass


def create_pipeline_model():
    pass


def create_ensemble_model(deploy_cfg, pipeline_cfg):
    inference_model_config = create_inference_model(deploy_cfg, pipeline_cfg)
    preprocess_model_config = create_preprocess_model()
    postprocess_model_config = create_postprocess_model()
    pipeline_model_config = create_pipeline_model()


def main():
    args = parse_args()
    model_path = args.model_path
    if not osp.isdir(model_path):
        model_path = osp.split(model_path)[-2]
    if osp.isdir(model_path):
        with open(osp.join(model_path, 'deploy.json'), 'r') as f:
            deploy_cfg = json.load(f)
        with open(osp.join(model_path, 'pipeline.json'), 'r') as f:
            pipeline_cfg = json.load(f)
    task = deploy_cfg['task']
    model_config = create_model_config('model', task, 'onnxruntime')
    data = serialize_model_config(model_config)
    print(data)


if __name__ == '__main__':
    main()

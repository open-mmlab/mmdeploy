# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from tritonclient.grpc import (InferenceServerClient, InferInput,
                               InferRequestedOutput)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='model name')
    parser.add_argument('image', type=str, help='image path')
    return parser.parse_args()


class GRPCTritonClient:

    def __init__(self, url, model_name, model_version):
        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._client = InferenceServerClient(self._url)
        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(
            self._model_name, self._model_version)
        print(f'[model config]:\n{model_config}')
        print(f'[model metadata]:\n{model_metadata}')
        self._inputs = {input.name: input for input in model_metadata.inputs}
        self._input_names = list(self._inputs)
        self._outputs = {
            output.name: output
            for output in model_metadata.outputs
        }
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]

    def infer(self, image):
        """
        Args:
            image: np.ndarray
        Returns:
            results: dict, {name : numpy.array}
        """

        inputs = [InferInput(self._input_names[0], image.shape, 'UINT8')]
        inputs[0].set_data_from_numpy(image)
        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=inputs,
            outputs=self._outputs_req)
        results = {name: results.as_numpy(name) for name in self._output_names}
        return results


def visualize(results):
    det_bboxes = results['bboxes']
    det_scores = results['scores']
    rec_texts = results['rec_texts']
    rec_scores = results['rec_scores']
    for i, (det_bbox, det_score, rec_text, rec_score) in \
            enumerate(zip(det_bboxes, det_scores, rec_texts, rec_scores)):
        print(f'bbox[{i}] ({det_bbox[0]:.2f}, {det_bbox[1]:.2f}), '
              f'({det_bbox[2]:.2f}, {det_bbox[3]:.2f}), ({det_bbox[4]:.2f}, '
              f'{det_bbox[5]:.2f}), ({det_bbox[6]:.2f}, {det_bbox[7]:.2f}), '
              f'{det_score[0]:.2f}')
        text = rec_text.decode('utf-8')
        print(f'text[{i}] {text}')


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    model_version = '1'
    url = 'localhost:8001'
    client = GRPCTritonClient(url, model_name, model_version)
    img = cv2.imread(args.image)
    results = client.infer(img)
    visualize(results)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from math import cos, sin

import cv2
import numpy as np
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


def visualize(img, results):
    bboxes = results['bboxes']
    labels = results['labels']
    for rbbox, label_id in zip(bboxes, labels):
        [cx, cy, w, h, angle], score = rbbox[0:5], rbbox[-1]
        if score < 0.1:
            continue
        [wx, wy, hx, hy] = \
            0.5 * np.array([w, w, -h, h]) * \
            np.array([cos(angle), sin(angle), sin(angle), cos(angle)])
        points = np.array([[[int(cx - wx - hx),
                             int(cy - wy - hy)],
                            [int(cx + wx - hx),
                             int(cy + wy - hy)],
                            [int(cx + wx + hx),
                             int(cy + wy + hy)],
                            [int(cx - wx + hx),
                             int(cy - wy + hy)]]])
        cv2.drawContours(img, points, -1, (0, 255, 0), 2)
    cv2.imwrite('oriented-object-detection.jpg', img)


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    model_version = '1'
    url = 'localhost:8001'
    client = GRPCTritonClient(url, model_name, model_version)
    img = cv2.imread(args.image)
    results = client.infer(img)
    visualize(img, results)

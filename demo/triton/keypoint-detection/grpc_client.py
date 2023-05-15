# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help='model name')
    parser.add_argument('image', type=str,
                        help='image path')
    return parser.parse_args()


class GRPCTritonClient:

    def __init__(self, url, model_name, model_version):
        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._client = InferenceServerClient(self._url)
        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(self._model_name,
                                                         self._model_version)
        print(f'[model config]:\n{model_config}')
        print(f'[model metadata]:\n{model_metadata}')
        self._inputs = {input.name: input for input in model_metadata.inputs}
        self._input_names = list(self._inputs)
        self._outputs = {
            output.name: output for output in model_metadata.outputs}
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]

    def infer(self, image, box):
        """
        Args:
            image: np.ndarray
            box: np.ndarray
        Returns:
            results: dict, {name : numpy.array}
        """

        inputs = [InferInput(self._input_names[0], image.shape,
                             "UINT8"),
                  InferInput(self._input_names[1], box.shape,
                             "BYTES")]
        inputs[0].set_data_from_numpy(image)
        inputs[1].set_data_from_numpy(box)
        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=inputs,
            outputs=self._outputs_req)
        results = {name: results.as_numpy(name) for name in self._output_names}
        return results


def visualize(img, results):
    batch_keypoints = results['keypoints']
    for keypoints in batch_keypoints:
        n = keypoints.shape[0]
        for i in range(n):
            x, y, score = keypoints[i]
            x, y = map(int, (x, y))
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)
    cv2.imwrite('keypoint-detection.jpg', img)


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    model_version = "1"
    url = "localhost:8001"
    client = GRPCTritonClient(url, model_name, model_version)
    img = cv2.imread(args.image)
    bbox = {
        'type': 'PoseBbox',
        'value': [
            {
                'bbox': [0.0, 0.0, img.shape[1], img.shape[0]]
            }
        ]
    }
    bbox = np.array([json.dumps(bbox).encode('utf-8')])
    results = client.infer(img, bbox)
    visualize(img, results)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math

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


def visualize(img, results):
    bboxes = results['bboxes']
    labels = results['labels']
    masks = results['masks']
    mask_offs = results['mask_offs']
    assert len(bboxes) == len(labels)
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        off, w, h = mask_offs[i]
        mask_data = masks[off:off + w * h]
        mask = mask_data.reshape(h, w)

        blue, green, red = cv2.split(img)
        x0 = int(max(math.floor(x1) - 1, 0))
        y0 = int(max(math.floor(y1) - 1, 0))
        mask_img = blue[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        cv2.bitwise_or(mask, mask_img, mask_img)
        img = cv2.merge([blue, green, red])

    cv2.imwrite('instance-segmentation.jpg', img)


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    model_version = '1'
    url = 'localhost:8001'
    client = GRPCTritonClient(url, model_name, model_version)
    img = cv2.imread(args.image)
    results = client.infer(img)
    visualize(img, results)

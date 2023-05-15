# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
                        help='model name')
    parser.add_argument('image', type=str,
                        help='image path')
    return parser.parse_args()


def get_palette(num_classes=256):
    state = np.random.get_state()
    # random color
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3))
    np.random.set_state(state)
    return [tuple(c) for c in palette]


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

    def infer(self, image):
        """
        Args:
            image: np.ndarray
        Returns:
            results: dict, {name : numpy.array}
        """

        inputs = [InferInput(self._input_names[0], image.shape,
                             "UINT8")]
        inputs[0].set_data_from_numpy(image)
        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=inputs,
            outputs=self._outputs_req)
        results = {name: results.as_numpy(name) for name in self._output_names}
        return results


def visualize(img, results):
    if 'mask' in results:
        seg = results['mask']
    else:
        score = results['score']
        seg = np.argmax(score, axis=0)

    palette = get_palette()
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    cv2.imwrite('semantic-segmentation.png', img)


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    model_version = "1"
    url = "localhost:8001"
    client = GRPCTritonClient(url, model_name, model_version)
    img = cv2.imread(args.image)
    results = client.infer(img)
    visualize(img, results)

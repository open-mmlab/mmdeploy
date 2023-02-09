# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from math import cos, sin

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model_name', type=str)
    parser.add_argument('image_path', type=str)
    parser.add_argument(
        '-v', '--model_version', type=str, required=False, default='1')
    parser.add_argument(
        '-u', '--url', type=str, required=False, default='localhost:8001')
    return parser.parse_args()


def get_palette(num_classes=256):
    state = np.random.get_state()
    # random color
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3))
    np.random.set_state(state)
    return [tuple(c) for c in palette]


def vis_cls(img, scores, label_ids):
    print('\n'.join(map(str, zip(scores, label_ids))))


def vis_det(img, bboxes, labels):
    for bbox, label in zip(bboxes, labels):
        (left, top, right, bottom), score = bbox[0:4].astype(int), bbox[4]
        if score < 0.3:
            continue
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
    return img


def vis_rdet(img, bboxes, labels):
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
    return img


def vis_seg(img, mask, scores):
    if mask is None:
        mask = np.argmax(scores, axis=0)

    palette = get_palette()
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[mask == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    return img.astype(np.uint8)


def vis_ocr(img, dets, text, text_score):
    pts = ((dets[:, 0:8] + 0.5).reshape(len(dets), -1, 2).astype(int))
    cv2.polylines(img, pts, True, (0, 255, 0), 2)
    print('\n'.join(map(str, zip(range(len(text)), text, text_score))))
    return img


def vis_pose(img, dets, kpts):
    pass


def main():
    args = parse_args()
    triton_client = grpcclient.InferenceServerClient(url=args.url)

    model_config = triton_client.get_model_config(
        model_name=args.model_name, model_version=args.model_version)

    img = cv2.imread(args.image_path)

    if img is None:
        print(f'failed to load image {args.image_path}')
        return

    task = model_config.config.parameters['task'].string_value

    task_map = dict(
        Classifier=(('scores', 'labels'), vis_cls),
        Detector=(('bboxes', 'labels'), vis_det),
        TextOCR=(('dets', 'text', 'text_score'), vis_ocr),
        Restorer=(('output',), lambda _, hires: hires),
        Segmentor=(('mask', 'score'), vis_seg),
        RotatedDetector=(('bboxes', 'labels'), None),
        DetPose=(('bboxes', 'keypoints'), vis_pose))

    output_names, visualize = task_map[task]

    # request input
    inputs = [grpcclient.InferInput('ori_img', img.shape, 'UINT8')]
    inputs[0].set_data_from_numpy(img)

    # request outputs
    outputs = map(grpcclient.InferRequestedOutput, output_names)

    # run inference
    response = triton_client.infer(
        model_config.config.name, inputs, outputs=list(outputs))

    # visualize results
    vis = visualize(img, *map(response.as_numpy, output_names))

    if vis is not None:
        cv2.imshow('', vis)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

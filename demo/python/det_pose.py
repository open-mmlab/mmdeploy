# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_runtime import Detector, PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use SDK Python API')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'det_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument(
        'pose_model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of input image')
    args = parser.parse_args()
    return args


def visualize(frame, keypoints, filename, thr=0.5, resize=1280):
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
               (255, 153, 255), (153, 204, 255), (255, 102, 255),
               (255, 51, 255), (102, 178, 255),
               (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
               (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
               (0, 0, 255), (255, 0, 0), (255, 255, 255)]
    link_color = [
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]
    point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

    scale = resize / max(frame.shape[0], frame.shape[1])

    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)

    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score in zip(keypoints, scores):
        show = [0] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
                show[u] = show[v] = 1
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    cv2.imwrite(filename, img)


def main():
    args = parse_args()

    # load image
    img = cv2.imread(args.image_path)

    # create object detector
    detector = Detector(
        model_path=args.det_model_path, device_name=args.device_name)
    # create pose detector
    pose_detector = PoseDetector(
        model_path=args.pose_model_path, device_name=args.device_name)

    # apply detector
    bboxes, labels, _ = detector(img)

    # filter detections
    keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
    bboxes = bboxes[keep, :4]

    # apply pose detector
    poses = pose_detector(img, bboxes)

    visualize(img, poses, 'det_pose_output.jpg', 0.5, 1280)


if __name__ == '__main__':
    main()

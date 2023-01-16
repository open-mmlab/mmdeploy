# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import cv2
from mmdeploy_python import PoseTracker


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
    parser.add_argument('video_path', help='path of input video')
    parser.add_argument(
        '--output_dir', help='output directory', default='output')
    args = parser.parse_args()
    return args


def visualize(frame, results, filename, thr=0.5, resize=1280):
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
    keypoints, bboxes, _ = results
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)
    bboxes *= scale
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score, bbox in zip(keypoints, scores, bboxes):
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

    video = cv2.VideoCapture(args.video_path)

    tracker = PoseTracker(
        det_model_path=args.det_model_path,
        pose_model_path=args.pose_model_path,
        device_name=args.device_name,
    )

    # optionally use OKS for keypoints similarity comparison
    coco_sigmas = [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ]
    state = tracker.CreateState(
        det_interval=1, det_min_bbox_size=100, keypoint_sigmas=coco_sigmas)

    os.makedirs(args.output_dir, exist_ok=True)

    frame_id = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        results = tracker(state, frame, detect=-1)
        visualize(frame, results,
                  f'{args.output_dir}/{str(frame_id).zfill(6)}.jpg')
        frame_id += 1


if __name__ == '__main__':
    main()

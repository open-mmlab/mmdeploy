# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
from mmdeploy_runtime import VideoRecognizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('video_path', help='path of an video')
    parser.add_argument(
        '--clip_len', help='Frames of each sampled output clip', default=1)
    parser.add_argument(
        '--frame_interval',
        help='Temporal interval of adjacent sampled frames.',
        default=1)
    parser.add_argument(
        '--num_clips', help='Number of clips to be sampled', default=25)
    args = parser.parse_args()
    return args


def SampleFrames(cap, clip_len, frame_interval, num_clips):
    if not cap.isOpened():
        print('failed to load video')
        exit(-1)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ori_clip_len = clip_len * frame_interval
    avg_interval = (num_frames - ori_clip_len + 1) / float(num_clips)
    frame_inds = []
    for i in range(num_clips):
        clip_offset = int(i * avg_interval + avg_interval / 2.0)
        for j in range(clip_len):
            ind = (j * frame_interval + clip_offset) % num_frames
            if num_frames <= ori_clip_len - 1:
                ind = j % num_frames
            frame_inds.append(ind)

    unique_inds = sorted(list(set(frame_inds)))
    buffer = {}
    ind = 0
    for i, tid in enumerate(unique_inds):
        while ind < tid:
            _, mat = cap.read()
            ind += 1
        _, mat = cap.read()
        buffer[tid] = mat
        ind += 1

    clips = []
    for tid in frame_inds:
        clips.append(buffer[tid])
    info = (clip_len, num_clips)
    return clips, info


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video_path)

    recognizer = VideoRecognizer(
        model_path=args.model_path, device_name=args.device_name, device_id=0)

    clips, info = SampleFrames(cap, args.clip_len, args.frame_interval,
                               args.num_clips)

    result = recognizer(clips, info)
    for label_id, score in result:
        print(label_id, score)


if __name__ == '__main__':
    main()

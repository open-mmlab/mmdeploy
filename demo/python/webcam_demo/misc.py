# Copyright (c) OpenMMLab. All rights reserved.
import time
from contextlib import contextmanager


def is_image(s: str):
    a = ['jpg', 'png', 'jfif', 'webp', 'jpeg']

    for t in a:
        if s.endswith('.' + t):
            return True

    return False


def is_video(s: str):
    a = ['mp4', 'flv']

    for t in a:
        if s.endswith('.' + t):
            return True

    return False


@contextmanager
def limit_max_fps(fps: float):
    t_start = time.time()
    try:
        yield
    finally:
        t_end = time.time()
        if fps is not None:
            t_sleep = 1.0 / fps - t_end + t_start
            if t_sleep > 0:
                time.sleep(t_sleep)

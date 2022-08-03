# Copyright (c) OpenMMLab. All rights reserved.
import logging
import threading
from abc import abstractclassmethod

import cv2
import mmdeploy_python
import numpy as np

from . import misc

# import misc


class AbstractWebcamDemo:

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.detector = mmdeploy_python.PoseDetector(model_path, device, 0)
        self.bbox = []

        self.cached_points = []

        self.visible = True
        self.help_visible = True

        self.cap = None
        self.writer = None

    def init_bbox(self, box):
        self.bbox = box

    # @abstractclassmethod
    def handle(self, frame):
        self.add_points(frame)
        self.add_help_message(frame)

    def get_points(self, frame):
        assert self.detector is not None
        assert self.bbox != []

        res = self.detector([frame], [[self.bbox]])[0]

        _, point_cnt, _ = res.shape
        self.cached_points = res[:, :, :2].reshape(point_cnt, 2).astype(int)

    # @abstractclassmethod
    def add_points(self, frame):
        if self.visible:
            for [x, y] in self.cached_points:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)

    def add_help_message(self, frame):
        return
        cv2.putText(frame, self.help_message,
                    cv2.getWindowImageRect('demo')[:2],
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5,
                    cv2.LINE_AA, False)

    def receive_key(self, key: int):
        if key == ord('q'):
            return True
        else:
            if key == ord('v'):
                self.visible = not self.visible
            elif key == ord('h'):
                self.help_visible = not self.help_visible

            return False

    @abstractclassmethod
    def run(self):
        pass


class ImageDemo(AbstractWebcamDemo):

    def __init__(self,
                 image_file: str,
                 model_path: str,
                 device: str = 'cuda',
                 output_file: str = ''):
        super().__init__(model_path, device)

        self.input_file = image_file
        self.output_file = output_file

    def handle(self, frame):
        self.get_points(frame)
        super().handle(frame)

    def run(self):
        img = cv2.imread(self.input_file)

        self.init_bbox([0, 0, *img.shape[:2]])
        self.handle(img)

        logging.info('Image as input')
        logging.info('Press any key to continue...')

        cv2.imshow('demo', img)
        cv2.imwrite('output.png', img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


class VideoDemo(AbstractWebcamDemo):

    def __init__(self,
                 video_file: str,
                 model_path: str,
                 device: str = 'cuda',
                 output_file: str = '',
                 code: str = ''):
        super().__init__(model_path, device)

        self.input_file = video_file
        self.output_file = output_file
        self.code = code

    def run(self):
        self.cap = cv2.VideoCapture(self.input_file)

        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frames.append(frame)

        self.cap.release()

        if not frames:
            logging.warning('No frame detected')
            # sys.exit(0)
        else:
            row, col, _ = frames[0].shape

            if self.output_file != '':
                self.writer = cv2.VideoWriter(
                    self.output_file, cv2.VideoWriter_fourcc(*self.code), 30,
                    (row, col))

            for frame in frames:
                self.handle(frame)

                cv2.imshow('demo', frame)
                if self.writer is not None:
                    self.writer.write(frame)

                if self.receive_key(cv2.waitKey(10)):
                    break

            if self.writer is not None:
                self.writer.release()

        logging.info('Procedure finished, press any key to continue...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class WebcamDemo(AbstractWebcamDemo):

    def __init__(self,
                 camera_id: int,
                 model_path: str,
                 device: str = 'cuda',
                 max_delay: float = 0.5,
                 max_fps=30,
                 skip: int = 2,
                 output_file: str = ''):
        super().__init__(model_path, device)

        self.camera_id = camera_id

        self.max_delay = max(0.0, max_delay)
        self.max_fps = max(5, max_fps)
        self.skip_count = max(skip, 1)

        assert output_file == '', 'Webcam output has not been supported'

        self.cnt = 0

    def work(self, frame, q):
        if self.cnt % self.skip_count == 0 or self.cached_points == []:
            self.get_points(frame)

        self.handle(frame)
        q.append(frame)

        self.cnt += 1

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)

        # writer = cv2.VideoWriter('output.avi',
        #   cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
        q = []

        flag = False
        last = None

        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)

        while True:
            with misc.limit_max_fps(self.max_fps):
                if not flag:
                    ret, frame = self.cap.read()

                    if not ret:
                        flag = True
                    else:
                        if not self.bbox:
                            self.init_bbox([0, 0, *frame.shape[:2]])

                        th = threading.Thread(
                            target=self.work, args=(frame, q))
                        th.setDaemon(True)
                        th.start()
                        # th.join()

                if len(q) > 0:
                    last = q[0]
                    q.pop(0)
                elif flag:
                    break

                if last is not None:
                    assert (type(last) == np.ndarray)

                    cv2.imshow('demo', last)
                    # writer.write(last)

                if self.receive_key(cv2.waitKey(10)):
                    break

        if self.cap is not None:
            self.cap.release()
        # writer.release()
        cv2.destroyAllWindows()

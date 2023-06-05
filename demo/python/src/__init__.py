# Copyright (c) OpenMMLab. All rights reserved.
import logging
import threading
from abc import abstractmethod

import cv2
import mmdeploy_python
import numpy as np

from . import misc, models

# import misc


class AbstractWebcamDemo:

    def __init__(self, detect_model_path: str, pose_model_path: str,
                 device: str):
        # print(f'pose_model_path = {pose_model_path}')
        if detect_model_path:
            self.detector = mmdeploy_python.Detector(detect_model_path, device,
                                                     0)
        else:
            self.detector = None

        self.pose_detector = mmdeploy_python.PoseDetector(
            pose_model_path, device, 0)
        self.pose_model = models.check_model(pose_model_path)

        self.boxes = []
        # self.labels = []
        # self.mask = []

        self.width = self.height = 0

        self.points = []

        self.visible = True
        self.help_visible = True

        self.cap = None
        self.writer = None

    def get_boxes(self, frame):
        if self.detector is None:
            self.boxes = [(0, 0, self.height, self.width)]
            return

        self.boxes = []

        boxes, labels, masks = self.detector([frame])[0]

        for index, box, label_id in zip([i for i in range(len(boxes))], boxes,
                                        labels):
            [left, top, right, bottom], score = box[0:4].astype(int), box[4]

            if score >= 0.3:
                self.boxes.append((left, top, right, bottom))

    def get_points(self, frame):
        assert self.pose_detector is not None
        # assert self.boxes != []

        self.points = []

        for box in self.boxes:
            res = self.pose_detector([frame], [[box]])[0]

            _, point_cnt, _ = res.shape
            self.points.append(res[:, :, :2].reshape(point_cnt, 2).astype(int))

    def update(self, frame):
        self.get_boxes(frame)
        self.get_points(frame)

    # @abstractmethod
    def handle(self, frame):
        if not self.width:
            self.width, self.height, _ = frame.shape

        self.draw_boxes(frame)
        self.draw_points(frame)
        self.draw_lines(frame)
        self.add_help_message(frame)

    def draw_boxes(self, frame):
        if self.detector is None:
            return

        for box in self.boxes:
            cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 0),
                thickness=1)

    def out_of_bound(self, x, y):
        return x < 0 or y < 0 or x > self.width or y > self.height

    # @abstractmethod
    def draw_points(self, frame):

        def draw(frame, x, y):
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 3)

        for vec in self.points:
            if self.visible:
                if self.pose_model in [models.DEEPPOSE_RESNET_152_RLE]:
                    for i in range(0, len(vec), 2):
                        x, y = vec[i]
                        draw(frame, x, y)
                else:
                    for [x, y] in vec:
                        draw(frame, x, y)

    def debug_points_id(self, frame):
        if self.visible:
            for vec in self.points:
                for i in range(len(vec)):
                    cv2.putText(frame, f'{i}', vec[i],
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                                2, cv2.LINE_AA, False)

    def draw_lines(self, frame):
        for vec in self.points:
            if self.visible and len(vec):
                # Sorry I have to hard-code
                if self.pose_model != models.UNKNOWN_MODEL:
                    if self.pose_model in [
                            models.DEEPPOSE_RESNET_50,
                            models.DEEPPOSE_RESNET_152
                    ]:
                        edges = [(4, 2), (2, 0), (0, 1), (1, 2), (1, 3),
                                 (7, 9), (5, 6), (5, 7), (6, 8), (8, 10),
                                 (5, 11), (6, 12), (11, 12), (11, 13),
                                 (13, 15), (12, 14), (14, 16)]

                    elif self.pose_model in [models.DEEPPOSE_RESNET_152_RLE]:
                        edges = [(8, 4), (4, 0), (0, 2), (2, 6), (12, 10),
                                 (12, 16), (10, 14)]

                    else:
                        assert False, \
                            'There must be some stupid error in the code.'

                else:
                    edges = []

                for i, j in edges:
                    cv2.line(frame, vec[i], vec[j], (0, 255, 0), thickness=1)

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

    @abstractmethod
    def run(self):
        pass


class ImageDemo(AbstractWebcamDemo):

    def __init__(self,
                 image_file: str,
                 detect_model_path: str,
                 pose_model_path: str,
                 device: str = 'cuda',
                 output_file: str = ''):
        super().__init__(detect_model_path, pose_model_path, device)

        self.input_file = image_file
        self.output_file = output_file

    def handle(self, frame):
        self.get_points(frame)
        super().handle(frame)

    def run(self):
        img = cv2.imread(self.input_file)

        # self.init_bbox([0, 0, *img.shape[:2]])
        self.handle(img)

        logging.info('Image as input')

        cv2.imwrite('output.png', img)

        while True:
            with misc.limit_max_fps(30):
                cv2.imshow('demo', img)

                if cv2.waitKey(10) == ord('q'):
                    break

        logging.info('Press any key to continue...')

        # cv2.waitKey(0)

        cv2.destroyAllWindows()


class VideoDemo(AbstractWebcamDemo):

    def __init__(self,
                 video_file: str,
                 detect_model_path: str,
                 pose_model_path: str,
                 device: str = 'cuda',
                 output_file: str = '',
                 code: str = ''):
        super().__init__(detect_model_path, pose_model_path, device)

        self.input_file = video_file
        self.output_file = output_file
        self.code = code

        self.paused = False

    def receive_key(self, key: int):
        if key == ord(' '):
            self.paused = not self.paused
            return False
        else:
            return super().receive_key(key)

    def run(self):
        self.cap = cv2.VideoCapture(self.input_file)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.output_file != '':
            cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)

        flag = False
        cnt = 0
        last = np.ndarray([])

        while True:
            with misc.limit_max_fps(self.fps):
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret or len(frame) == 0:
                        break
                    else:
                        last = frame.copy()
                else:
                    frame = last.copy()

                # if self.bbox == []:
                # 	self.init_bbox([0, 0, row, col])

                if self.writer is None and self.output_file != '':
                    row, col, _ = frame.shape
                    self.writer = cv2.VideoWriter(
                        self.output_file, cv2.VideoWriter_fourcc(*'XVID'), 30,
                        (col, row))

                # self.get_points(frame)
                self.update(frame)
                self.handle(frame)

                if self.writer is not None:
                    self.writer.write(frame.astype(np.uint8))
                    cnt += 1
                    print(f'cnt = {cnt}')
                else:
                    cv2.imshow('demo', frame)
                    if self.receive_key(cv2.waitKey(10)):
                        break

                flag = True

        self.cap.release()

        if self.writer is not None:
            self.writer.release()

        if not flag:
            logging.warning('No frame detected')
            # sys.exit(0)
        else:
            logging.info('Procedure finished.')

        # cv2.waitKey(0)
        cv2.destroyAllWindows()


class WebcamDemo(AbstractWebcamDemo):

    def __init__(self,
                 camera_id: int,
                 detect_model_path: str,
                 pose_model_path: str,
                 device: str = 'cuda',
                 max_delay: float = 0.5,
                 max_fps=30,
                 skip: int = 2,
                 output_file: str = ''):
        super().__init__(detect_model_path, pose_model_path, device)

        self.camera_id = camera_id

        self.max_delay = max(0.0, max_delay)
        self.max_fps = max(5, max_fps)
        self.skip_count = max(skip, 1)

        assert output_file == '', 'Webcam output has not been supported'

        self.cnt = 0

    def work(self, frame, q):
        if self.cnt % self.skip_count == 0 or self.points == []:
            self.update(frame)

        self.handle(frame)
        q.append(frame)

        self.cnt += 1

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)

        # writer = cv2.VideoWriter('output.avi', \
        #   cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
        q = []

        flag = False
        last = None

        cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)

        while True:
            with misc.limit_max_fps(self.max_fps):
                if not flag:
                    ret, frame = self.cap.read()

                    if not ret:
                        flag = True
                    else:
                        # if not self.bbox:
                        # 	self.init_bbox([0, 0, *frame.shape[:2]])
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

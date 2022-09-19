import random
from pathlib import Path

import cv2
import numpy as np
import torch

from mmdeploy.backend.tensorrt import TRTWrapper


def letterbox(im,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=False,
              scaleup=True,
              stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_copy = im.copy()
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)  # add border
    return im, im_copy, r, (dw, dh)


names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
colors = {
    name: [random.randint(0, 255) for _ in range(3)]
    for i, name in enumerate(names)
}

cv2.setNumThreads(0)
device = torch.device('cuda:0')
engine_file = 'work_dir_trt_yolov5/end2end.engine'
model = TRTWrapper(engine_file)
image_path = Path('data/coco/train2017')

for i in image_path.glob('*.jpg'):

    image = cv2.imread(str(i))
    image, image_orin, ratio, dwdh = letterbox(image, auto=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_Copy = image.copy()

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = torch.from_numpy(image.astype(np.float32))
    im /= 255

    inputs = dict(input=im.to(device))
    outputs = model(inputs)

    # 后处理放这里
    outputs['dets'][:, :, :4] -= torch.tensor([dwdh * 2],
                                              device=device,
                                              dtype=torch.float32)

    for (x0, y0, x1, y1, conf), cls in zip(outputs['dets'][0],
                                           outputs['labels'][0]):
        name = names[int(cls)]
        color = colors[name]
        cv2.rectangle(image_orin, [int(x0), int(y0)],
                      [int(x1), int(y1)], color, 2)
        cv2.putText(
            image_orin,
            name, (int(x0), int(y0) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, [225, 255, 255],
            thickness=2)

    cv2.imshow('win', image_orin)
    cv2.waitKey(0)

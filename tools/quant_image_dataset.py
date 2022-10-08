# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import mmcv
from mmengine import Config, FileClient
from torch.utils.data import Dataset

from mmdeploy.apis import build_task_processor


class QuantizationImageDataset(Dataset):

    def __init__(
        self,
        path: str,
        deploy_cfg: Config,
        model_cfg: Config,
        file_client_args: Optional[dict] = None,
        extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                                     '.pgm', '.tif'),
    ):
        super().__init__()
        task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
        self.task_processor = task_processor

        self.samples = []
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client = FileClient.infer_client(file_client_args, path)
        self.path = path

        assert self.file_client.isdir(path)
        files = list(
            self.file_client.list_dir_or_file(
                path,
                list_dir=False,
                list_file=True,
                recursive=False,
            ))
        for file in files:
            if self.is_valid_file(self.file_client.join_path(file)):
                path = self.file_client.join_path(self.path, file)
                self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = mmcv.imread(sample)
        data = self.task_processor.create_input(image)
        return data[0]

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import mmcv
from mmcv import FileClient
from torch.utils.data import Dataset

from mmdeploy.utils import Codebase, get_codebase


class QuantizationImageDataset(Dataset):

    def __init__(
        self,
        path: str,
        deploy_cfg: mmcv.Config,
        model_cfg: mmcv.Config,
        file_client_args: Optional[dict] = None,
        extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                                     '.pgm', '.tif'),
    ):
        super().__init__()
        codebase_type = get_codebase(deploy_cfg)
        self.exclude_pipe = ['LoadImageFromFile']
        if codebase_type == Codebase.MMCLS:
            from mmcls.datasets.pipelines import Compose
        elif codebase_type == Codebase.MMDET:
            from mmdet.datasets.pipelines import Compose
        elif codebase_type == Codebase.MMDET3D:
            from mmdet3d.datasets.pipelines import Compose
            self.exclude_pipe.extend([
                'LoadMultiViewImageFromFiles', 'LoadImageFromFileMono3D',
                'LoadPointsFromMultiSweeps', 'LoadPointsFromDict'
            ])
        elif codebase_type == Codebase.MMEDIT:
            from mmedit.datasets.pipelines import Compose
            self.exclude_pipe.extend(
                ['LoadImageFromFileList', 'LoadPairedImageFromFile'])
        elif codebase_type == Codebase.MMOCR:
            from mmocr.datasets.pipelines import Compose
            self.exclude_pipe.extend(
                ['LoadImageFromNdarray', 'LoadImageFromLMDB'])
        elif codebase_type == Codebase.MMPOSE:
            from mmpose.datasets.pipelines import Compose
        elif codebase_type == Codebase.MMROTATE:
            from mmrotate.datasets.pipelines import Compose
        elif codebase_type == Codebase.MMSEG:
            from mmseg.datasets.pipelines import Compose
        else:
            raise Exception(
                'Not supported codebase_type {}'.format(codebase_type))
        pipeline = filter(lambda val: val['type'] not in self.exclude_pipe,
                          model_cfg.data.test.pipeline)

        self.preprocess = Compose(list(pipeline))
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
        data = dict(img=image)
        data = self.preprocess(data)
        from mmcv.parallel import collate
        data = collate([data], samples_per_gpu=1)

        return {'img': data['img'].squeeze()}

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

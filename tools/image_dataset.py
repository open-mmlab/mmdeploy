# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import DataLoader
from mmcv import FileClient
import mmcv
from typing import Optional, Sequence

class QuantizationImageDataset(DataLoader):
    def __init__(self, data_prefix: str, processor, 
                shape: tuple,
                file_client_args: Optional[dict] = None,
                extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),):
        super(QuantizationImageDataset).__init__()
        self.processor = processor
        self.samples = []
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client = FileClient.infer_client(file_client_args, data_prefix)
        files = list(file_client.list_dir_or_file(
                data_prefix,
                list_dir=False,
                list_file=True,
                recursive=False,
            ))
        for file in sorted(list(files)):
            if self.is_valid_file(file):
                path = file_client.join_path(file)
                samples.append(path)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        ret = []
        for index in (iter_start, iter_end):
            file = self.samples[index]
            image = mmcv.imread(file)
            tensor = self.processor.create_input(image, shape)
            ret.append(tensor)
        return ret
        
    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

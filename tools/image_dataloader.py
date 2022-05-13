# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import DataLoader, get_worker_info
from mmcv import FileClient
import mmcv
from typing import Optional, Sequence

class QuantizationImageDataloader(DataLoader):
    def __init__(self, data_prefix: str, processor, 
                file_client_args: Optional[dict] = None,
                extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),):
        super(QuantizationImageDataset).__init__()
        self.processor = processor
        self.samples = []
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client = FileClient.infer_client(file_client_args, data_prefix)
        self.data_prefix = data_prefix
            
        assert self.file_client.isdir(data_prefix)
        files = list(self.file_client.list_dir_or_file(
                data_prefix,
                list_dir=False,
                list_file=True,
                recursive=False,
            ))
        for file in files:
            if self.is_valid_file(self.file_client.join_path(file)):
                path = self.file_client.join_path(self.data_prefix, file)
                self.samples.append(path)

    def __iter__(self):
        return self
    
    def __next__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.samples)
        else:
            per_worker = int(math.ceil(len(self.samples) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(per_worker, len(self.samples))
        for index in range(iter_start, iter_end):
            sample = self.samples[index]
            image = mmcv.imread(sample)
            yield self.processor.create_input(image)[0]
    # def __iter__(self):
    #     import pdb
    #     pdb.set_trace()
    #     worker_info = get_worker_info()
    #     if worker_info is None:
    #         iter_start = 0
    #         iter_end = len(self.samples)
    #     else:
    #         per_worker = int(math.ceil(len(self.samples) / float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = worker_id * per_worker
    #         iter_end = min(per_worker, len(self.samples))
    #     ret = []
    #     for index in range(iter_start, iter_end):
    #         sample = self.samples[index]
    #         image = mmcv.imread(sample)
    #         input = self.processor.create_input(image)
    #         ret.append(input[0])
    #     return ret
        
    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

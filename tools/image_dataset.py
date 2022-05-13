# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import Dataset, get_worker_info
from mmcv import FileClient
import mmcv
from typing import Optional, Sequence

class QuantizationImageDataset(Dataset):
    def __init__(self, path: str, processor, 
                file_client_args: Optional[dict] = None,
                extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),):
        super(QuantizationImageDataset).__init__()
        self.processor = processor
        self.samples = []
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.file_client = FileClient.infer_client(file_client_args, path)
        self.path = path
            
        assert self.file_client.isdir(path)
        files = list(self.file_client.list_dir_or_file(
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
        input =  self.processor.create_input(image)
        return {'img': input[1][0].squeeze()}
        
    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

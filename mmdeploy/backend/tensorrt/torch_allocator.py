# Copyright (c) OpenMMLab. All rights reserved.
import tensorrt as trt
import torch

from mmdeploy.utils import get_root_logger


class TorchAllocator(trt.IGpuAllocator):
    """PyTorch Cuda Allocator Wrapper."""

    def __init__(self, device_id: int = 0) -> None:
        super().__init__()

        self.device_id = device_id
        self.mems = set()
        self.caching_delete = torch._C._cuda_cudaCachingAllocator_raw_delete

    def __del__(self):
        """destructor."""
        mems = self.mems.copy()
        (self.deallocate(mem) for mem in mems)

    def allocate(self: trt.IGpuAllocator, size: int, alignment: int,
                 flags: int) -> int:
        """allocate gpu memory.

        Args:
            self (trt.IGpuAllocator): gpu allocator
            size (int): memory size.
            alignment (int): memory alignment.
            flags (int): flags.

        Returns:
            int: memory address.
        """
        torch_stream = torch.cuda.current_stream(self.device_id)
        logger = get_root_logger()
        logger.debug(f'allocate {size} memory with TorchAllocator.')
        assert alignment >= 0
        if alignment > 0:
            size = size | (alignment - 1) + 1
        mem = torch.cuda.caching_allocator_alloc(
            size, device=self.device_id, stream=torch_stream)
        self.mems.add(mem)
        return mem

    def deallocate(self: trt.IGpuAllocator, memory: int) -> bool:
        """deallocate memory.

        Args:
            self (trt.IGpuAllocator): gpu allocator
            memory (int): memory address.

        Returns:
            bool: deallocate success.
        """
        if memory not in self.mems:
            return False

        self.caching_delete(memory)
        self.mems.discard(memory)
        return True

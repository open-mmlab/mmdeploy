# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, Sequence

import numpy as np
import pycuda.autoinit  # noqa:F401
import pycuda.driver as cuda
import tensorrt as trt

DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2


class IteratorCalibrator(trt.IInt8Calibrator):
    """HDF5 calibrator.

    Args:
        calib_file (str | h5py.File):  Input calibration file.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        model_type (str): Input model type, defaults to 'end2end'.
        device_id (int): Cuda device id, defaults to 0.
        algorithm (trt.CalibrationAlgoType): Calibration algo type, defaults
            to `trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2`.
    """

    def __init__(
            self,
            data_iter: Iterable,
            device_id: int = 0,
            algorithm: trt.CalibrationAlgoType = DEFAULT_CALIBRATION_ALGORITHM,
            **kwargs):
        super().__init__()
        self._data_generator = data_iter

        self._device_id = device_id
        self._algorithm = algorithm

        # create buffers that will hold data batches
        self._buffers = dict()

        next_data = next(self._data_generator)
        names = list(next_data.keys())
        self._batch_size = next_data[names[0]].shape[0]
        self._next_data = next_data

    def __del__(self):
        """Close h5py file if necessary."""
        del self._data_generator

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""

        if self._next_data is not None:
            # host to device
            ret = []

            for name in names:
                data_np = self._next_data[name]

                is_torch_data = False
                try:
                    import torch
                    if isinstance(data_np, torch.Tensor):
                        is_torch_data = True
                except Exception:
                    pass

                if is_torch_data:
                    data_np = data_np.cuda(self._device_id)
                    self._buffers[name] = data_np
                    ret.append(data_np.data_ptr())
                else:
                    assert isinstance(data_np, np.ndarray)
                    data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                    cuda.memcpy_htod(data_np_cuda_ptr,
                                     np.ascontiguousarray(data_np))
                    self._buffers[name] = data_np_cuda_ptr
                    ret.append(data_np_cuda_ptr)
            try:
                self._next_data = next(self._data_generator)
            except StopIteration:
                self._next_data = None

            return ret
        else:
            return None

    def get_algorithm(self) -> trt.CalibrationAlgoType:
        """Get Calibration algo type.

        Returns:
            trt.CalibrationAlgoType: Calibration algo type.
        """
        return self._algorithm

    def get_batch_size(self) -> int:
        """Get batch size.

        Returns:
            int: An integer represents batch size.
        """
        return self._batch_size

    def read_calibration_cache(self, *args, **kwargs):
        """Read calibration cache.

        Notes:
            No need to implement this function.
        """
        pass

    def write_calibration_cache(self, cache, *args, **kwargs):
        """Write calibration cache.

        Notes:
            No need to implement this function.
        """
        pass

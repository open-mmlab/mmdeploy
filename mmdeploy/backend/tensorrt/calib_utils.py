# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Union

import h5py
import numpy as np
import pycuda.autoinit  # noqa:F401
import pycuda.driver as cuda
import tensorrt as trt

DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2


class HDF5Calibrator(trt.IInt8Calibrator):
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
            calib_file: Union[str, h5py.File],
            input_shapes: Dict[str, Sequence[int]],
            model_type: str = 'end2end',
            device_id: int = 0,
            algorithm: trt.CalibrationAlgoType = DEFAULT_CALIBRATION_ALGORITHM,
            **kwargs):
        super().__init__()

        if isinstance(calib_file, str):
            calib_file = h5py.File(calib_file, mode='r')

        assert 'calib_data' in calib_file
        calib_data = calib_file['calib_data']
        assert model_type in calib_data
        calib_data = calib_data[model_type]

        self.calib_file = calib_file
        self.calib_data = calib_data
        self.device_id = device_id
        self.algorithm = algorithm
        self.input_shapes = input_shapes
        self.kwargs = kwargs

        # create buffers that will hold data batches
        self.buffers = dict()

        self.count = 0
        first_input_group = calib_data[list(calib_data.keys())[0]]
        self.dataset_length = len(first_input_group)
        self.batch_size = first_input_group['0'].shape[0]

    def __del__(self):
        """Close h5py file if necessary."""
        if hasattr(self, 'calib_file'):
            self.calib_file.close()

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:

            ret = []
            for name in names:
                input_group = self.calib_data[name]
                data_np = input_group[str(self.count)][...].astype(np.float32)

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None

    def get_algorithm(self) -> trt.CalibrationAlgoType:
        """Get Calibration algo type.

        Returns:
            trt.CalibrationAlgoType: Calibration algo type.
        """
        return self.algorithm

    def get_batch_size(self) -> int:
        """Get batch size.

        Returns:
            int: An integer represents batch size.
        """
        return self.batch_size

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

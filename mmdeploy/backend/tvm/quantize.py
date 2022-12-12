# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Sequence, Union

import numpy as np
import tvm
from tvm.runtime.ndarray import array


class HDF5Dataset:
    """HDF5 dataset.

    Args:
        calib_file (str | h5py.File):  Input calibration file.
        input_shapes (Dict[str, Sequence[int]]): The shape of
            each input.
        model_type (str): Input model type, defaults to 'end2end'.
        device (str): Device type, default to llvm.
    """

    def __init__(
        self,
        calib_file: Union[str, Any],
        input_shapes: Dict[str, Sequence[int]],
        model_type: str = 'end2end',
        device: str = 'llvm',
    ) -> None:
        import h5py
        if isinstance(calib_file, str):
            calib_file = h5py.File(calib_file, mode='r')

        assert 'calib_data' in calib_file
        calib_data = calib_file['calib_data']
        assert model_type in calib_data
        calib_data = calib_data[model_type]

        self.calib_file = calib_file
        self.calib_data = calib_data
        self.device = device
        self.input_shapes = input_shapes

        first_input_group = calib_data[list(calib_data.keys())[0]]
        self.dataset_length = len(first_input_group)

    def __call__(self):
        """Create dataset generator.

        Yields:
            Iterator[Any]: data in the dataset
        """
        for idx in range(self.dataset_length):

            ret = dict()
            for name, opt_shape in self.input_shapes.items():
                input_group = self.calib_data[name]
                data_np = input_group[str(idx)][...].astype(np.float32)

                data_shape = data_np.shape

                # tile the input data
                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_nd = array(data_np, tvm.device(self.device))

                ret[name] = data_nd
            yield ret

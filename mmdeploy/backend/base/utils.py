# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Sequence, Union


def get_obj_by_qualname(qualname: str) -> Any:
    """Get object by the qualname.

    Args:
        qualname (str): The qualname of the object

    Returns:
        Any: The object with qualname
    """
    split_qualname = qualname.split('.')
    for i in range(len(split_qualname), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_qualname[:i])))
            break
        except Exception:
            continue

    obj = eval(qualname)

    return obj


def create_h5pydata_generator(data_file: Union[str, Any],
                              input_shapes: Dict[str, Sequence[int]],
                              data_type: str = 'end2end'):
    """Create data generator for h5py data.

    Args:
        data_file (Union[str, Any]): h5py file.
        input_shapes (Dict[str, Sequence]): Input shape of each input tensors.
        data_type (str, optional): Data type id. Defaults to 'end2end'.
    """
    import h5py
    import numpy as np
    if isinstance(data_file, str):
        data_file = h5py.File(data_file, mode='r')

    try:
        assert 'calib_data' in data_file
        calib_data = data_file['calib_data']
        assert data_type in calib_data
        calib_data = calib_data[data_type]

        names = list(calib_data.keys())
        first_input_group = calib_data[list(calib_data.keys())[0]]
        dataset_length = len(first_input_group)

        # iterate over all data
        for idx in range(dataset_length):

            yield_data = dict()
            for name in names:
                input_group = calib_data[name]
                data_np = input_group[str(idx)][...]

                # tile the tensor so we can keep the same distribute
                opt_shape = input_shapes[name]
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                yield_data[name] = data_np

            yield yield_data

    except Exception as e:
        raise e
    finally:
        data_file.close()

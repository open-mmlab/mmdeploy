# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any


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

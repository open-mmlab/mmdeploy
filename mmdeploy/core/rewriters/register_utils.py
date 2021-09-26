from typing import Any


def eval_with_import(path: str) -> Any:
    """Evaluate the string as Python script.

    Args:
        path (str): The path to evaluate.

    Returns:
        Any: The result of evaluate.
    """
    split_path = path.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue
    return eval(path)

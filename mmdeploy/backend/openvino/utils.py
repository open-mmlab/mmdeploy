# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union


class ModelOptimizerOptions:
    """A class to make it easier to support additional arguments for the Model
    Optimizer that can be passed through the deployment configuration.

    Example:
        >>> deploy_cfg = load_config(deploy_cfg_path)
        >>> mo_options = deploy_cfg.get('mo_options', None)
        >>> mo_options = ModelOptimizerOptions(mo_options)
        >>> mo_args = mo_options.get_options()
    """

    def __init__(self,
                 mo_options: Optional[Dict[str, Union[Dict, List]]] = None):
        self.args = ''
        self.flags = ''
        if mo_options is not None:
            self.args = self.__parse_args(mo_options)
            self.flags = self.__parse_flags(mo_options)

    def __parse_args(self, mo_options: Dict[str, Union[Dict, List]]) -> str:
        """Parses a dictionary with arguments into a string."""
        mo_args_str = ''
        if 'args' in mo_options:
            for key, value in mo_options['args'].items():
                value_str = f'"{value}"' if isinstance(value, list) else value
                mo_args_str += f'{key}={value_str} '
        return mo_args_str

    def __parse_flags(self, mo_options: Dict[str, Union[Dict, List]]) -> str:
        """Parses a list with flags into a string."""
        mo_flags_str = ''
        if 'flags' in mo_options:
            mo_flags_str += ' '.join(mo_options['flags'])
        return mo_flags_str

    def get_options(self) -> str:
        """Returns a string with additional arguments for the Model Optimizer.

        If there are no additional arguments, it will return an empty string.
        """
        return self.args + self.flags

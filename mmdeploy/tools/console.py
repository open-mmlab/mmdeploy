# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import sys
from argparse import ArgumentParser
from typing import Iterable, List, Optional


def import_custom_modules(custom_modules: Iterable):
    """Import custom module."""
    from mmdeploy.utils import get_root_logger
    logger = get_root_logger(0)
    custom_modules = [] if custom_modules is None else custom_modules

    for qualname in custom_modules:
        try:
            importlib.import_module(qualname)
            logger.info(f'Import custom module: {qualname}')
        except Exception as e:
            logger.warning('Failed to import custom module: '
                           f'{qualname} with error: {e}')


def list_command(parser: ArgumentParser,
                 input_args: Optional[List[str]] = None):
    """List command."""
    parser.description = 'List available backend and task.'
    arg_group = parser.add_argument_group('List Options')
    arg_group.add_argument(
        '--backend', action='store_true', help='List available backend.')
    arg_group.add_argument(
        '--task', action='store_true', help='List available task.')
    arg_group.add_argument(
        '--custom-modules', type=str, nargs='*', help='Custom module path.')
    args = parser.parse_args(input_args)

    def _print_pretty_table(data_list, title):
        """print pretty table."""
        max_data_length = [len(t) for t in title]

        for data in data_list:
            assert len(title) == len(data)
            for idx, d in enumerate(data):
                max_data_length[idx] = max(max_data_length[idx], len(d))

        # print title
        title_all = ' '.join([
            f'{t}' + (' ' * (length - len(t)))
            for length, t in zip(max_data_length, title)
        ])
        print(title_all)

        # print dash
        dash_all = ' '.join(['-' * length for length in max_data_length])
        print(dash_all)

        # print data
        for data in data_list:
            data_all = ' '.join([
                f'{t}' + (' ' * (length - len(t)))
                for length, t in zip(max_data_length, data)
            ])
            print(data_all)

    # custom import
    import_custom_modules(args.custom_modules)
    enable_list_backend = args.backend
    enable_list_task = args.task

    if not enable_list_backend and not enable_list_task:
        enable_list_task = True
        enable_list_backend = True

    if enable_list_backend:
        from mmdeploy.backend.base import get_backend_manager
        from mmdeploy.utils import Backend

        exclude_backend_lists = [Backend.DEFAULT, Backend.PYTORCH, Backend.SDK]
        backend_lists = [
            backend.value for backend in Backend
            if backend not in exclude_backend_lists
        ]

        # get all available backend
        available_backend = []
        for backend in backend_lists:
            backend_mgr = get_backend_manager(backend)
            if backend_mgr.is_available():
                try:
                    available_backend.append(
                        (backend, backend_mgr.get_version()))
                except Exception as e:
                    sys.stderr.write(f'List backend: {backend} failed.'
                                     f' with error: {e}\n')

        _print_pretty_table(available_backend, ['Backend', 'Version'])

    if enable_list_task:
        # TODO: add list task here
        pass


def show_command(parser: ArgumentParser,
                 input_args: Optional[List[str]] = None):
    """Show command."""
    parser.description = 'Show environment of the backend or task.'
    arg_group = parser.add_argument_group('Show Options')
    arg_group.add_argument('name', help='The object name to show.')
    arg_group.add_argument(
        '--custom-modules', type=str, nargs='*', help='Custom module path.')
    args = parser.parse_args(input_args)

    # custom import
    import_custom_modules(args.custom_modules)

    obj_name = args.name

    # Check if obj is backend
    from mmdeploy.utils import Backend

    exclude_backend_lists = [Backend.DEFAULT, Backend.PYTORCH, Backend.SDK]
    backend_lists = [
        backend.value for backend in Backend
        if backend not in exclude_backend_lists
    ]

    if obj_name in backend_lists:
        from mmdeploy.backend.base import get_backend_manager
        backend_mgr = get_backend_manager(obj_name)
        if backend_mgr.is_available():
            backend_mgr.check_env(print)
        else:
            sys.stderr.write(f'Backend: {obj_name} is not available.\n')

    # TODO: add show task here


def run_command(parser: ArgumentParser,
                input_args: Optional[List[str]] = None):
    """Run command."""

    # extract help
    help = False
    if '-h' in input_args:
        help = True
        input_args.remove('-h')
    if '--help' in input_args:
        help = True
        input_args.remove('--help')

    parser.description = 'Run console tools of backend or task.'
    arg_group = parser.add_argument_group('Run Options')
    arg_group.add_argument(
        'obj_name',
        default=None,
        help='The backend or task name to run the console tools.')
    arg_group.add_argument(
        '--custom-modules', type=str, nargs='*', help='Custom module path.')

    if help:
        if len(input_args) == 0 or input_args[0] == '--custom-modules':
            parser.print_help()
            exit()

    args, remain_args = parser.parse_known_args(input_args)

    # custom import
    import_custom_modules(args.custom_modules)

    obj_name = args.obj_name

    if help:
        remain_args += ['--help']

    # Check if obj is backend
    from mmdeploy.utils import Backend

    exclude_backend_lists = [Backend.DEFAULT, Backend.PYTORCH, Backend.SDK]
    backend_lists = [
        backend.value for backend in Backend
        if backend not in exclude_backend_lists
    ]

    if obj_name in backend_lists:
        from mmdeploy.backend.base import get_backend_manager
        backend_mgr = get_backend_manager(obj_name)
        if backend_mgr.is_available():
            parser = ArgumentParser()
            try:
                generator = backend_mgr.parse_args(parser, remain_args)
                next(generator)
                next(generator)
            except StopIteration:
                print('Run finish.')
            except NotImplementedError:
                sys.stderr.write(
                    f'Backend: {obj_name} run has not been implemented.\n')
        else:
            sys.stderr.write(f'Backend: {obj_name} is not available.\n')

    # TODO: add run task here

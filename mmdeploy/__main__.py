# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys as _sys

if __name__ == '__main__':
    # args default to the system args
    console_args = _sys.argv[1:]

    # extract help
    help = False
    if '-h' in console_args:
        help = True
        console_args.remove('-h')
    if '--help' in console_args:
        help = True
        console_args.remove('--help')

    # add root parser
    parser = argparse.ArgumentParser(
        'mmdeploy', description='MMDeploy Toolkit')
    command_parsers = parser.add_subparsers(title='Commands', dest='command')
    list_parser = command_parsers.add_parser(
        'list', help='List available backend and task.')
    show_parser = command_parsers.add_parser(
        'show', help='Should information about the object.')
    run_parser = command_parsers.add_parser(
        'run', help='Run console tools of backend or task.')
    args, remain_args = parser.parse_known_args(console_args)

    # parse command
    command = getattr(args, 'command', None)

    if help:
        remain_args = ['--help'] + remain_args
    if command == 'list':
        from mmdeploy.tools.console import list_command
        list_command(list_parser, remain_args)
    elif command == 'show':
        from mmdeploy.tools.console import show_command
        show_command(list_parser, remain_args)
    elif command == 'run':
        from mmdeploy.tools.console import run_command
        run_command(list_parser, remain_args)
    else:
        parser.print_help()
        parser.exit()

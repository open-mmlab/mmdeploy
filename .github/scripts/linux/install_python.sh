#!/bin/bash

if [ $# -lt 1 ]; then
  echo 'use python 3.8.5 as default'
  PYTHON_VERSION=3.8.5
else
  PYTHON_VERSION=$1
fi

sudo apt-get update
# liblzma-dev need to be installed. Refer to https://github.com/pytorch/vision/issues/2921
# python3-tk tk-dev is for 'import tkinter'
sudo apt-get install -y liblzma-dev python3-tk tk-dev
# python3+ need to be reinstalled due to https://github.com/pytorch/vision/issues/2921
pyenv uninstall -f "$PYTHON_VERSION"
pyenv install "$PYTHON_VERSION"
pyenv global "$PYTHON_VERSION"

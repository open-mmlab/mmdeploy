#!/bin/bash

python ./tools/package_tools/mmdeploy_builder.py tools/package_tools/config/linux-x64.yml \
    --backend onnxruntime .
mv mmdeploy-*-onnxruntime* /prebuild-mmdeploy
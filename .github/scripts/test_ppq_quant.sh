#!/bin/sh

set -e

# print env
python tools/check_env.py
wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/dataset.tar
tar -xvf dataset.tar
export work_dir=work_dir
mkdir -p $work_dir
python -m mim download mmcls --config resnet18_8xb32_in1k --dest $work_dir
export model_cfg=$work_dir/resnet18_8xb32_in1k.py
export checkpoint=$work_dir/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
export deploy_cfg=configs/mmcls/classification_ncnn-int8_static.py
python tools/torch2onnx.py $deploy_cfg $model_cfg $checkpoint tests/data/tiger.jpeg \
  --work-dir $work_dir \
  --device cpu
ls -lah $work_dir
python tools/onnx2ncnn_quant_table.py \
  --onnx $work_dir/end2end.onnx \
  --deploy-cfg $deploy_cfg \
  --model-cfg $model_cfg \
  --out-onnx $work_dir/quant.onnx \
  --out-table $work_dir/ncnn.table \
  --image-dir dataset
ls -lah $work_dir

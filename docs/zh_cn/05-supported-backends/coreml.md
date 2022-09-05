# Core ML 支持情况

目前 mmdeploy 集成了 OpenMMLab 算法库中 Pytorch 模型到 Core ML模型的转换以及推理。

## 安装

转换 mmdet 中的模型，需要编译 libtorch 支持 nms 等自定义算子

```bash
cd ${PYTORCH_DIR}
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=`which python` \
    -DCMAKE_INSTALL_PREFIX=install \
    -DDISABLE_SVE=ON # 低版本比如1.8.0需要加上这个参数
make install
```

## 使用

```bash
python tools/deploy.py \
    configs/mmdet/detection/detection_coreml_static-800x1344.py \
    /mmdetection_dir/configs/retinanet/retinanet_r18_fpn_1x_coco.py \
    /checkpoint/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth \
    /mmdetection_dir/demo/demo.jpg \
    --work-dir work_dir/retinanet \
    --device cpu \
    --dump-info
```

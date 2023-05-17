# Object detection serving

## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs/retinanet/retinanet_r18_fpn_1x_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r18_fpn_1x_coco/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth  \
    ../mmdetection/demo/demo.jpg \
    --work-dir work_dir/retinanet \
    --dump-info \
    --device cuda
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/retinanet \
    /model-repository
```

## Start triton server
```
tritonserver --model-repository=/model-repository
```

## Run client code output container
```
python3 demo/triton/object-detection/grpc_client.py \
    model \
    /path/to/image
```
# Oriented object detection serving

## Starting a docker container

```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model

```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmrotate/rotated-detection_tensorrt_dynamic-320x320-1024x1024.py \
    ../mmrotate/configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
    https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
    ../mmrotate/demo/demo.jpg \
    --dump-info \
    --work-dir work_dir/rrcnn \
    --device cuda
```

## Convert tensorrt model to triton format

```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/rrcnn \
    /model-repository
```

## Start triton server

```
tritonserver --model-repository=/model-repository
```

## Run client code output container

```
python3 demo/triton/oriented-object-detection/grpc_client.py \
    model \
    /path/to/image
```

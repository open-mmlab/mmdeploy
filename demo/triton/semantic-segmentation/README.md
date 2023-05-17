# Semantic segmentation serving

## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmseg/segmentation_tensorrt-fp16_static-512x1024.py \
    ../mmsegmentation/configs/pspnet/pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py \
    https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth \
    ../mmsegmentation/demo/demo.png \
    --work-dir work_dir/pspnet \
    --dump-info \
    --device cuda
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/pspnet \
    /model-repository
```

## Start triton server
```
tritonserver --model-repository=/model-repository
```

## Run client code output container
```
python3 demo/triton/semantic-segmentation/grpc_client.py \
    model \
    /path/to/image
```
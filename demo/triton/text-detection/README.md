# Text detection serving

## Starting a docker container

```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model

```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmocr/text-detection/text-detection_tensorrt_dynamic-320x320-2240x2240.py \
    ../mmocr/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015.py \
    https://download.openmmlab.com/mmocr/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth \
    ../mmocr/demo/demo_text_det.jpg \
    --work-dir work_dir/panet \
    --dump-info \
    --device cuda:0
```

## Convert tensorrt model to triton format

```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/panet \
    /model-repository
```

## Start triton server

```
tritonserver --model-repository=/model-repository
```

## Run client code output container

```
python3 demo/triton/text-detection/grpc_client.py \
    model \
    /path/to/image
```

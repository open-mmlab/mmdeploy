# Image classification serving


## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmpretrain/classification_tensorrt_static-224x224.py \
    ../mmpretrain/configs/resnet/resnet18_8xb32_in1k.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
    ../mmpretrain/demo/demo.JPEG \
    --device cuda \
    --work-dir work_dir/resnet \
    --dump-info
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/resnet \
    /model-repository
```

## Start triton server
```
tritonserver --model-repository=/model-repository
```

## Run client code output container
```
python3 demo/triton/image-classification/grpc_client.py \
    model \
    /path/to/image
```

# Text recognition serving

## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_tensorrt-fp16_dynamic-1x32x32-1x32x640.py \
    ../mmocr/configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py \
    https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_mini-vgg_5e_mj/crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth \
    ../mmocr/demo/demo_text_recog.jpg \
    --work-dir work_dir/crnn \
    --device cuda \
    --dump-info
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/crnn \
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

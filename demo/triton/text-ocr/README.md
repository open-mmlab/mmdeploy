# Text ocr serving

## Starting a docker container

```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model

```
cd /root/workspace/mmdeploy

# text-detection
python3 tools/deploy.py \
    configs/mmocr/text-detection/text-detection_tensorrt_dynamic-320x320-2240x2240.py \
    ../mmocr/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015.py \
    https://download.openmmlab.com/mmocr/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth \
    ../mmocr/demo/demo_text_det.jpg \
    --work-dir work_dir/panet \
    --dump-info \
    --device cuda:0

# text-recognition
python3 tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_tensorrt-fp16_dynamic-1x32x32-1x32x640.py \
    ../mmocr/configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py \
    https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_mini-vgg_5e_mj/crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth \
    ../mmocr/demo/demo_text_recog.jpg \
    --work-dir work_dir/crnn \
    --device cuda \
    --dump-info
```

## Ensemble detection and recognition model

```
cd /root/workspace/mmdeploy
cp -r demo/triton/text-ocr/serving /model-repository
cp -r work_dir/panet/* /model-repository/model/1/text_detection/
cp -r work_dir/crnn/* /model-repository/model/1/text_recognition/
```

## Start triton server

```
tritonserver --model-repository=/model-repository
```

## Run client code output container

```
python3 demo/triton/text-ocr/grpc_client.py \
    model \
    /path/to/image
```

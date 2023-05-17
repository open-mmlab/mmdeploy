# Keypoint detection serving

## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmpose/pose-detection_tensorrt_static-256x192.py \
    ../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    demo/resources/human-pose.jpg \
    --work-dir work_dir/hrnet \
    --dump-info \
    --device cuda
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/hrnet \
    /model-repository
```

## Start triton server
```
tritonserver --model-repository=/model-repository
```

## Run client code output container
```
python3 demo/triton/keypoint-detection/grpc_client.py \
    model \
    /path/to/image
```
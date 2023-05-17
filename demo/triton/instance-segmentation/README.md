# Instance segmentation serving


## Starting a docker container
```
docker run -it --rm --gpus all openmmlab/mmdeploy:triton-22.12
```

## Convert pytorch model to tensorrt model
```
cd /root/workspace/mmdeploy
python3 tools/deploy.py \
    configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth \
    ../mmdetection/demo/demo.jpg \
    --work-dir work_dir/maskrcnn \
    --dump-info \
    --device cuda
```

## Convert tensorrt model to triton format
```
cd /root/workspace/mmdeploy
python3 demo/triton/to_triton_model.py \
    /root/workspace/mmdeploy/work_dir/maskrcnn \
    /model-repository
```

## Start triton server
```
tritonserver --model-repository=/model-repository
```

## Run client code output container
```
python3 demo/triton/instance-segmentation/grpc_client.py \
    model \
    /path/to/image
```

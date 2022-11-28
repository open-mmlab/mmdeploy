# Test on embedded device

Here are the test conclusions of our edge devices. You can directly obtain the results of your own environment with [model profiling](../02-how-to-run/profile_model.md).

## Software and hardware environment

- host OS ubuntu 18.04
- backend SNPE-1.59
- device Mi11 (qcom 888)

## mmcls

|                                                              model                                                               |   dataset   | spatial | fp32 top-1 (%) | snpe gpu hybrid fp32 top-1 (%) | latency (ms) |
| :------------------------------------------------------------------------------------------------------------------------------: | :---------: | :-----: | :------------: | :----------------------------: | :----------: |
| [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | ImageNet-1k | 224x224 |     69.55      |            69.83\*             |     20±7     |
|    [MobilenetV2](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)     | ImageNet-1k | 224x224 |     71.86      |            72.14\*             |     15±6     |

tips:

1. The ImageNet-1k dataset is too large to test, only part of the dataset is used (8000/50000)
2. The heating of device will downgrade the frequency, so the time consumption will actually fluctuate. Here are the stable values after running for a period of time. This result is closer to the actual demand.

## mmocr detection

|                                                       model                                                       |  dataset  | spatial  | fp32 hmean | snpe gpu hybrid hmean | latency(ms) |
| :---------------------------------------------------------------------------------------------------------------: | :-------: | :------: | :--------: | :-------------------: | :---------: |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) | ICDAR2015 | 1312x736 |   0.795    |    0.785 @thr=0.9     |  3100±100   |

## mmpose

|                                                                               model                                                                               |  dataset   | spatial | snpe hybrid AR@IoU=0.50 | snpe hybrid AP@IoU=0.50 | latency(ms) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :-----: | :---------------------: | :---------------------: | :---------: |
| [pose_hrnet_w32](https://github.com/open-mmlab/mmpose/blob/master/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py) | Animalpose | 256x256 |          0.997          |          0.989          |   630±50    |

tips:

- Test `pose_hrnet` using AnimalPose's test dataset instead of val dataset.

## mmseg

|                                                       model                                                       |  dataset   | spatial  | mIoU  | latency(ms) |
| :---------------------------------------------------------------------------------------------------------------: | :--------: | :------: | :---: | :---------: |
| [fcn](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn/fcn_r18-d8_512x1024_80k_cityscapes.py) | Cityscapes | 512x1024 | 71.11 |  4915±500   |

tips:

- `fcn` works fine with 512x1024 size. Cityscapes dataset uses 1024x2048 resolution which causes device to reboot.

## Notes

- We needs to manually split the mmdet model into two parts. Because
  - In snpe source code, `onnx_to_ir.py` can only parse onnx input while `ir_to_dlc.py` does not support `topk` operator
  - UDO (User Defined Operator) does not work with `snpe-onnx-to-dlc`
- mmedit model
  - `srcnn` requires cubic resize which snpe does not support
  - `esrgan` converts fine, but loading the model causes the device to reboot
- mmrotate depends on [e2cnn](https://pypi.org/project/e2cnn/) and needs to be installed manually [its Python3.6 compatible branch](https://github.com/QUVA-Lab/e2cnn)

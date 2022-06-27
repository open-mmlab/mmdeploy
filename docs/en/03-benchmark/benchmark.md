# Benchmark

## Backends

CPU: ncnn, ONNXRuntime, OpenVINO

GPU: ncnn, TensorRT, PPLNN

## Latency benchmark

### Platform

- Ubuntu 18.04
- ncnn 20211208
- Cuda 11.3
- TensorRT 7.2.3.4
- Docker 20.10.8
- NVIDIA tesla T4 tensor core GPU for TensorRT

### Other settings

- Static graph
- Batch size 1
- Synchronize devices after each inference.
- We count the average inference performance of 100 images of the dataset.
- Warm up. For ncnn, we warm up 30 iters for all codebases. As for other backends: for classification, we warm up 1010 iters; for other codebases, we warm up 10 iters.
- Input resolution varies for different datasets of different codebases. All inputs are real images except for `mmediting` because the dataset is not large enough.

Users can directly test the speed through [model profiling](../02-how-to-run/how_to_measure_performance_of_models.md). And here is the benchmark in our environment.

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmcls</th>
    <th align="center" colspan="5">TensorRT(ms)</th>
    <th align="center" colspan="2">PPLNN(ms)</th>
    <th align="center" colspan="2">ncnn(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" colspan="1" rowspan="2">model</td>
    <td align="center" colspan="1" rowspan="2">spatial</td>
    <td align="center" colspan="3">T4</td>
    <td align="center" colspan="2">JetsonNano2GB</td>
    <td align="center" colspan="1">Jetson TX2</td>
    <td align="center" colspan="1">T4</td>
    <td align="center" colspan="1">SnapDragon888</td>
    <td align="center" colspan="1">Adreno660</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">int8</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp32</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet/resnet50_b32x8_imagenet.py"> ResNet </a></td>
    <td align="center">224x224</td>
    <td align="center">2.97</td>
    <td align="center">1.26</td>
    <td align="center">1.21</td>
    <td align="center">59.32</td>
    <td align="center">30.54</td>
    <td align="center">24.13</td>
    <td align="center">1.30</td>
    <td align="center">33.91</td>
    <td align="center">25.93</td>
  </tr>
  <tr>
    <td align="center"> <a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py"> ResNeXt </a></td>
    <td align="center">224x224</td>
    <td align="center">4.31</td>
    <td align="center">1.42</td>
    <td align="center">1.37</td>
    <td align="center">88.10</td>
    <td align="center">49.18</td>
    <td align="center">37.45</td>
    <td align="center">1.36</td>
    <td align="center">133.44</td>
    <td align="center">69.38</td>
  </tr>
  <tr>
    <td align="center"> <a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet/seresnet50_b32x8_imagenet.py">  SE-ResNet </a></td>
    <td align="center">224x224</td>
    <td align="center">3.41</td>
    <td align="center">1.66</td>
    <td align="center">1.51</td>
    <td align="center">74.59</td>
    <td align="center">48.78</td>
    <td align="center">29.62</td>
    <td align="center">1.91</td>
    <td align="center">107.84</td>
    <td align="center">80.85</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py">  ShuffleNetV2 </a></td>
    <td align="center">224x224</td>
    <td align="center">1.37</td>
    <td align="center">1.19</td>
    <td align="center">1.13</td>
    <td align="center">15.26</td>
    <td align="center">10.23</td>
    <td align="center">7.37</td>
    <td align="center">4.69</td>
    <td align="center">9.55</td>
    <td align="center">10.66</td>
  </tr>
</tbody>
</table>
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmdet part1</th>
    <th align="center" colspan="4">TensorRT(ms)</th>
    <th align="center" colspan="1">PPLNN(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="2" colspan="1">model</td>
    <td align="center" rowspan="2" colspan="1">spatial</td>
    <td align="center" colspan="3">T4</td>
    <td align="center" colspan="1">Jetson TX2</td>
    <td align="center" colspan="1">T4</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">int8</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py">YOLOv3</a></td>
    <td align="center">320x320</td>
    <td align="center">14.76</td>
    <td align="center">24.92</td>
    <td align="center">24.92</td>
    <td align="center">-</td>
    <td align="center">18.07</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py">SSD-Lite</a></td>
    <td align="center">320x320</td>
    <td align="center">8.84</td>
    <td align="center">9.21</td>
    <td align="center">8.04</td>
    <td align="center">1.28</td>
    <td align="center">19.72</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py">RetinaNet</a></td>
    <td align="center">800x1344</td>
    <td align="center">97.09</td>
    <td align="center">25.79</td>
    <td align="center">16.88</td>
    <td align="center">780.48</td>
    <td align="center">38.34</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py">FCOS</a></td>
    <td align="center">800x1344</td>
    <td align="center">84.06</td>
    <td align="center">23.15</td>
    <td align="center">17.68</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py">FSAF</a></td>
    <td align="center">800x1344</td>
    <td align="center">82.96</td>
    <td align="center">21.02</td>
    <td align="center">13.50</td>
    <td align="center">-</td>
    <td align="center">30.41</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py">Faster R-CNN</a></td>
    <td align="center">800x1344</td>
    <td align="center">88.08</td>
    <td align="center">26.52</td>
    <td align="center">19.14</td>
    <td align="center">733.81</td>
    <td align="center">65.40</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py">Mask R-CNN</a></td>
    <td align="center">800x1344</td>
    <td align="center">104.83</td>
    <td align="center">58.27</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">86.80</td>
  </tr>
</tbody>
</table>
</div>

<div style="margin-left: 25px;">
<table>
<thead>
  <tr>
    <th align="center" colspan="2">mmdet part2</th>
    <th align="center" colspan="2">ncnn</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="2">model</td>
    <td align="center" rowspan="2">spatial</td>
    <td align="center" colspan="1">SnapDragon888</td>
    <td align="center" colspan="1">Adreno660</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp32</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py">MobileNetv2-YOLOv3</a></td>
    <td align="center">320x320</td>
    <td align="center">48.57</td>
    <td align="center">66.55</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py">SSD-Lite</a></td>
    <td align="center">320x320</td>
    <td align="center">44.91</td>
    <td align="center">66.19</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_tiny_8x8_300e_coco.py">YOLOX</a></td>
    <td align="center">416x416</td>
    <td align="center">111.60</td>
    <td align="center">134.50</td>
  </tr>
</tbody>
</table>
</div>

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmedit</th>
    <th align="center" colspan="4">TensorRT(ms)</th>
    <th align="center" colspan="1">PPLNN(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="2">model</td>
    <td align="center" rowspan="2">spatial</td>
    <td align="center" colspan="3">T4</td>
    <td align="center" colspan="1">Jetson TX2</td>
    <td align="center" colspan="1">T4</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">int8</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py">ESRGAN</a></td>
    <td align="center">32x32</td>
    <td align="center">12.64</td>
    <td align="center">12.42</td>
    <td align="center">12.45</td>
    <td align="center">-</td>
    <td align="center">7.67</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py">SRCNN</a></td>
    <td align="center">32x32</td>
    <td align="center">0.70</td>
    <td align="center">0.35</td>
    <td align="center">0.26</td>
    <td align="center">58.86</td>
    <td align="center">0.56</td>
  </tr>
</tbody>
</table>
</div>

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmocr</th>
    <th align="center" colspan="3">TensorRT(ms)</th>
    <th align="center" colspan="1">PPLNN(ms)</th>
    <th align="center" colspan="2">ncnn(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="2">model</td>
    <td align="center" rowspan="2">spatial</td>
    <td align="center" colspan="3">T4</td>
    <td align="center" colspan="1">T4</td>
    <td align="center" colspan="1">SnapDragon888</td>
    <td align="center" colspan="1">Adreno660</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">int8</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp32</td>
  </tr>
    <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py">DBNet</a></td>
    <td align="center">640x640</td>
    <td align="center">10.70</td>
    <td align="center">5.62</td>
    <td align="center">5.00</td>
    <td align="center">34.84</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/crnn/crnn_academic_dataset.py">CRNN</a></td>
    <td align="center">32x32</td>
    <td align="center">1.93 </td>
    <td align="center">1.40</td>
    <td align="center">1.36</td>
    <td align="center">-</td>
    <td align="center">10.57</td>
    <td align="center">20.00</td>
</tbody>
</table>
</div>

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmseg</th>
    <th align="center" colspan="4">TensorRT(ms)</th>
    <th align="center" colspan="1">PPLNN(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="2">model</td>
    <td align="center" rowspan="2">spatial</td>
    <td align="center" colspan="3">T4</td>
    <td align="center" colspan="1">Jetson TX2</td>
    <td align="center" colspan="1">T4</td>
  </tr>
  <tr>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
    <td align="center" colspan="1">int8</td>
    <td align="center" colspan="1">fp32</td>
    <td align="center" colspan="1">fp16</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py">FCN</a></td>
    <td align="center">512x1024</td>
    <td align="center">128.42</td>
    <td align="center">23.97</td>
    <td align="center">18.13</td>
    <td align="center">1682.54</td>
    <td align="center">27.00</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py">PSPNet</a></td>
    <td align="center">1x3x512x1024</td>
    <td align="center">119.77</td>
    <td align="center">24.10</td>
    <td align="center">16.33</td>
    <td align="center">1586.19</td>
    <td align="center">27.26</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py">DeepLabV3</a></td>
    <td align="center">512x1024</td>
    <td align="center">226.75</td>
    <td align="center">31.80</td>
    <td align="center">19.85</td>
    <td align="center">-</td>
    <td align="center">36.01</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py">DeepLabV3+</a></td>
    <td align="center">512x1024</td>
    <td align="center">151.25</td>
    <td align="center">47.03</td>
    <td align="center">50.38</td>
    <td align="center">2534.96</td>
    <td align="center">34.80</td>
  </tr>
</tbody>
</table>
</div>

## Performance benchmark

Users can directly test the performance through [how_to_evaluate_a_model.md](../02-how-to-run/how_to_evaluate_a_model.md). And here is the benchmark in our environment.

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">mmcls</th>
    <th align="center">PyTorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNX Runtime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet/resnet18_b32x8_imagenet.py">ResNet-18</a></td>
    <td align="center">top-1</td>
    <td align="center">69.90</td>
    <td align="center">69.90</td>
    <td align="center">69.88</td>
    <td align="center">69.88</td>
    <td align="center">69.86</td>
    <td align="center">69.86</td>
    <td align="center">69.86</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">89.43</td>
    <td align="center">89.43</td>
    <td align="center">89.34</td>
    <td align="center">89.34</td>
    <td align="center">89.33</td>
    <td align="center">89.38</td>
    <td align="center">89.34</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py">ResNeXt-50</a></td>
    <td align="center">top-1</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">-</td>
    <td align="center">77.78</td>
    <td align="center">77.89</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">-</td>
    <td align="center">93.64</td>
    <td align="center">93.65</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py">SE-ResNet-50</a></td>
    <td align="center">top-1</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.75</td>
    <td align="center">77.63</td>
    <td align="center">77.73</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.83</td>
    <td align="center">93.72</td>
    <td align="center">93.84</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py">ShuffleNetV1 1.0x</a></td>
    <td align="center">top-1</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">67.71</td>
    <td align="center">68.11</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.58</td>
    <td align="center">87.80</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py">ShuffleNetV2 1.0x</a></td>
    <td align="center">top-1</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.54</td>
    <td align="center">69.10</td>
    <td align="center">69.54</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.91</td>
    <td align="center">88.58</td>
    <td align="center">88.92</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py">MobileNet V2</a></td>
    <td align="center">top-1</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.87</td>
    <td align="center">70.91</td>
    <td align="center">71.84</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.40</td>
    <td align="center">89.85</td>
    <td align="center">90.41</td>
  </tr>
</tbody>
</table>
</div>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">mmdet</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">task</td>
    <td align="center">dataset</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py">YOLOV3</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">33.7</td>
    <td align="center">33.7</td>
    <td align="center">-</td>
    <td align="center">33.5</td>
    <td align="center">33.5</td>
    <td align="center">33.5</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py">SSD</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">25.5</td>
    <td align="center">25.5</td>
    <td align="center">-</td>
    <td align="center">25.5</td>
    <td align="center">25.5</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py">RetinaNet</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">36.5</td>
    <td align="center">36.4</td>
    <td align="center">-</td>
    <td align="center">36.4</td>
    <td align="center">36.4</td>
    <td align="center">36.3</td>
    <td align="center">36.5</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py">FCOS</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">36.6</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">36.6</td>
    <td align="center">36.5</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py">FSAF</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">37.4</td>
    <td align="center">37.4</td>
    <td align="center">-</td>
    <td align="center">37.4</td>
    <td align="center">37.4</td>
    <td align="center">37.2</td>
    <td align="center">37.4</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_s_8x8_300e_coco.py">YOLOX</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">40.5</td>
    <td align="center">40.3</td>
    <td align="center">-</td>
    <td align="center">40.3</td>
    <td align="center">40.3</td>
    <td align="center">29.3</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py">Faster R-CNN</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">37.4</td>
    <td align="center">37.3</td>
    <td align="center">-</td>
    <td align="center">37.3</td>
    <td align="center">37.3</td>
    <td align="center">37.1</td>
    <td align="center">37.3</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/atss/atss_r50_fpn_1x_coco.py">ATSS</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">39.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">39.4</td>
    <td align="center">39.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py">Cascade R-CNN</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">40.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">40.4</td>
    <td align="center">40.4</td>
    <td align="center">-</td>
    <td align="center">40.4</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_1x_coco.py">GFL</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">40.2</td>
    <td align="center">-</td>
    <td align="center">40.2</td>
    <td align="center">40.2</td>
    <td align="center">40.0</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py">RepPoints</a></td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">37.0</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">36.9</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py">Mask R-CNN</a></td>
    <td align="center" rowspan="2">Instance Segmentation</td>
    <td align="center" rowspan="2">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">38.2</td>
    <td align="center">38.1</td>
    <td align="center">-</td>
    <td align="center">38.1</td>
    <td align="center">38.1</td>
    <td align="center">-</td>
    <td align="center">38.0</td>
  </tr>
  <tr>
    <td align="center">mask AP</td>
    <td align="center">34.7</td>
    <td align="center">34.7</td>
    <td align="center">-</td>
    <td align="center">33.7</td>
    <td align="center">33.7</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
</div>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">mmedit</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNX Runtime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">task</td>
    <td align="center">dataset</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py">SRCNN</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.4316</td>
    <td align="center">28.4120</td>
    <td align="center">28.4323</td>
    <td align="center">28.4323</td>
    <td align="center">28.4286</td>
    <td align="center">28.1995</td>
    <td align="center">28.4311</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8099</td>
    <td align="center">0.8106</td>
    <td align="center">0.8097</td>
    <td align="center">0.8097</td>
    <td align="center">0.8096</td>
    <td align="center">0.7934</td>
    <td align="center">0.8096</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py">ESRGAN</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.2700</td>
    <td align="center">28.2619</td>
    <td align="center">28.2592</td>
    <td align="center">28.2592</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">28.2624</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.7778</td>
    <td align="center">0.7784</td>
    <td align="center">0.7764</td>
    <td align="center">0.7774</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.7765</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py">ESRGAN-PSNR</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.6428</td>
    <td align="center">30.6306</td>
    <td align="center">30.6444</td>
    <td align="center">30.6430</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.0426</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8559</td>
    <td align="center">0.8565</td>
    <td align="center">0.8558</td>
    <td align="center">0.8558</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8557</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.py">SRGAN</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">27.9499</td>
    <td align="center">27.9252</td>
    <td align="center">27.9408</td>
    <td align="center">27.9408</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.9388</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.7846</td>
    <td align="center">0.7851</td>
    <td align="center">0.7839</td>
    <td align="center">0.7839</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.7839</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py">SRResNet</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.2252</td>
    <td align="center">30.2069</td>
    <td align="center">30.2300</td>
    <td align="center">30.2300</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">30.2294</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8491</td>
    <td align="center">0.8497</td>
    <td align="center">0.8488</td>
    <td align="center">0.8488</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8488</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py">Real-ESRNet</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.0297</td>
    <td align="center">-</td>
    <td align="center">27.7016</td>
    <td align="center">27.7016</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.7049</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8236</td>
    <td align="center">-</td>
    <td align="center">0.8122</td>
    <td align="center">0.8122</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8123</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py">EDSR</a></td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.2223</td>
    <td align="center">30.2192</td>
    <td align="center">30.2214</td>
    <td align="center">30.2214</td>
    <td align="center">30.2211</td>
    <td align="center">30.1383</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8500</td>
    <td align="center">0.8507</td>
    <td align="center">0.8497</td>
    <td align="center">0.8497</td>
    <td align="center">0.8497</td>
    <td align="center">0.8469</td>
    <td align="center"> - </td>
  </tr>
</tbody>
</table>
</div>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">mmocr</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">task</td>
    <td align="center">dataset</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
  </tr>
  <tr>
    <td align="center" rowspan="3"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py">DBNet*</a></td>
    <td align="center" rowspan="3">TextDetection</td>
    <td align="center" rowspan="3">ICDAR2015</td>
    <td align="center">recall</td>
    <td align="center">0.7310</td>
    <td align="center">0.7308</td>
    <td align="center">0.7304</td>
    <td align="center">0.7198</td>
    <td align="center">0.7179</td>
    <td align="center">0.7111</td>
    <td align="center">0.7304</td>
    <td align="center">0.7309</td>
  </tr>
  <tr>
    <td align="center">precision</td>
    <td align="center">0.8714</td>
    <td align="center">0.8718</td>
    <td align="center">0.8714</td>
    <td align="center">0.8677</td>
    <td align="center">0.8674</td>
    <td align="center">0.8688</td>
    <td align="center">0.8718</td>
    <td align="center">0.8714</td>
  </tr>
  <tr>
    <td align="center">hmean</td>
    <td align="center">0.7950</td>
    <td align="center">0.7949</td>
    <td align="center">0.7950</td>
    <td align="center">0.7868</td>
    <td align="center">0.7856</td>
    <td align="center">0.7821</td>
    <td align="center">0.7949</td>
    <td align="center">0.7950</td>
  </tr>
  <tr>
    <td align="center" rowspan="3"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py">PSENet</a></td>
    <td align="center" rowspan="3">TextDetection</td>
    <td align="center" rowspan="3">ICDAR2015</td>
    <td align="center">recall</td>
    <td align="center">0.7526</td>
    <td align="center">0.7526</td>
    <td align="center">0.7526</td>
    <td align="center">0.7526</td>
    <td align="center">0.7520</td>
    <td align="center">0.7496</td>
    <td align="center">-</td>
    <td align="center">0.7526</td>
  </tr>
  <tr>
    <td align="center">precision</td>
    <td align="center">0.8669</td>
    <td align="center">0.8669</td>
    <td align="center">0.8669</td>
    <td align="center">0.8669</td>
    <td align="center">0.8668</td>
    <td align="center">0.8550</td>
    <td align="center">-</td>
    <td align="center">0.8669</td>
  </tr>
  <tr>
    <td align="center">hmean</td>
    <td align="center">0.8057</td>
    <td align="center">0.8057</td>
    <td align="center">0.8057</td>
    <td align="center">0.8057</td>
    <td align="center">0.8054</td>
    <td align="center">0.7989</td>
    <td align="center">-</td>
    <td align="center">0.8057</td>
  </tr>
  <tr>
    <td align="center" rowspan="3"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py">PANet</a></td>
    <td align="center" rowspan="3">TextDetection</td>
    <td align="center" rowspan="3">ICDAR2015</td>
    <td align="center">recall</td>
    <td align="center">0.7401</td>
    <td align="center">0.7401</td>
    <td align="center">0.7401</td>
    <td align="center">0.7357</td>
    <td align="center">0.7366</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.7401</td>
  </tr>
  <tr>
    <td align="center">precision</td>
    <td align="center">0.8601</td>
    <td align="center">0.8601</td>
    <td align="center">0.8601</td>
    <td align="center">0.8570</td>
    <td align="center">0.8586</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.8601</td>
  </tr>
  <tr>
    <td align="center">hmean</td>
    <td align="center">0.7955</td>
    <td align="center">0.7955</td>
    <td align="center">0.7955</td>
    <td align="center">0.7917</td>
    <td align="center">0.7930</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.7955</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/crnn/crnn_academic_dataset.py">CRNN</a></td>
    <td align="center">TextRecognition</td>
    <td align="center">IIIT5K</td>
    <td align="center">acc</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8063</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py">SAR</a></td>
    <td align="center">TextRecognition</td>
    <td align="center">IIIT5K</td>
    <td align="center">acc</td>
    <td align="center">0.9517</td>
    <td align="center">-</td>
    <td align="center">0.9287</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmocr/tree/master/configs/textrecog/satrn/satrn_small.py">SATRN</a></td>
    <td align="center">TextRecognition</td>
    <td align="center">IIIT5K</td>
    <td align="center">acc</td>
    <td align="center">0.9470</td>
    <td align="center">0.9487</td>
    <td align="center">0.9487</td>
    <td align="center">0.9487</td>
    <td align="center">0.9483</td>
    <td align="center">0.9483</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
</div>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">mmseg</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">dataset</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py">FCN</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">72.25</td>
    <td align="center">72.36</td>
    <td align="center">-</td>
    <td align="center">72.36</td>
    <td align="center">72.35</td>
    <td align="center">74.19</td>
    <td align="center">72.35</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py">PSPNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">78.55</td>
    <td align="center">78.66</td>
    <td align="center">-</td>
    <td align="center">78.26</td>
    <td align="center">78.24</td>
    <td align="center">77.97</td>
    <td align="center">78.09</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py">deeplabv3</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.09</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">79.12</td>
    <td align="center">79.12</td>
    <td align="center">78.96</td>
    <td align="center">79.12</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py">deeplabv3+</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.61</td>
    <td align="center">79.60</td>
    <td align="center">-</td>
    <td align="center">79.60</td>
    <td align="center">79.60</td>
    <td align="center">79.43</td>
    <td align="center">79.60</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py">Fast-SCNN</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">70.96</td>
    <td align="center">70.96</td>
    <td align="center">-</td>
    <td align="center">70.93</td>
    <td align="center">70.92</td>
    <td align="center">66.00</td>
    <td align="center">70.92</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py">UNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">69.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">69.10</td>
    <td align="center">69.10</td>
    <td align="center">68.95</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann/ann_r50-d8_512x1024_40k_cityscapes.py">ANN</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.40</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.32</td>
    <td align="center">77.32</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py">APCNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.40</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.32</td>
    <td align="center">77.32</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py">BiSeNetV1</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.44</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">74.44</td>
    <td align="center">74.43</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py">BiSeNetV2</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">73.21</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">73.21</td>
    <td align="center">73.21</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/cgnet/cgnet_512x1024_60k_cityscapes.py">CGNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">68.25</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">68.27</td>
    <td align="center">68.27</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/emanet/emanet_r50-d8_512x1024_80k_cityscapes.py">EMANet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.59</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.59</td>
    <td align="center">77.6</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet/encnet_r50-d8_512x1024_40k_cityscapes.py">EncNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">75.67</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">75.66</td>
    <td align="center">75.66</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes.py">ERFNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">71.08</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">71.08</td>
    <td align="center">71.07</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes.py">FastFCN</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">79.12</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/gcnet/gcnet_r50-d8_512x1024_40k_cityscapes.py">GCNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.69</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.69</td>
    <td align="center">77.69</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/icnet/icnet_r18-d8_832x832_80k_cityscapes.py">ICNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">76.29</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">76.36</td>
    <td align="center">76.36</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/isanet/isanet_r50-d8_512x1024_40k_cityscapes.py">ISANet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">78.49</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">78.49</td>
    <td align="center">78.49</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes.py">OCRNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.30</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">73.66</td>
    <td align="center">73.67</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/point_rend/pointrend_r50_512x1024_80k_cityscapes.py">PointRend</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">76.47</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">76.41</td>
    <td align="center">76.42</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/sem_fpn/fpn_r50_512x1024_80k_cityscapes.py">Semantic FPN</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.52</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">74.52</td>
    <td align="center">74.52</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes.py">STDC</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">75.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">75.10</td>
    <td align="center">75.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes.py">STDC</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.17</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.17</td>
    <td align="center">77.17</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py">UPerNet</a></td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.19</td>
    <td align="center">77.18</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k.py">Segmenter</a></td>
    <td align="center">ADE20K</td>
    <td align="center">mIoU</td>
    <td align="center">44.32</td>
    <td align="center">44.29</td>
    <td align="center">44.29</td>
    <td align="center">44.29</td>
    <td align="center">43.34</td>
    <td align="center">43.35</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
</div>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">mmpose</th>
    <th align="center">Pytorch</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="2">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">task</td>
    <td align="center">dataset</td>
    <td align="center">metric</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py">HRNet</a></td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">-</td>
    <td align="center">0.748</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">-</td>
    <td align="center">0.802</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py">LiteHRNet</a></td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.663</td>
    <td align="center">0.663</td>
    <td align="center">0.663</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.663</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.728</td>
    <td align="center">0.728</td>
    <td align="center">0.728</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.728</td>
  </tr>
  <tr>
    <td align="center" rowspan="2"><a href="https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py">MSPN</a></td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">-</td>
    <td align="center">0.762</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">-</td>
    <td align="center">0.825</td>
  </tr>
</tbody>
</table>
</div>

<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">mmrotate</th>
    <th align="center">Pytorch</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="2">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">model</td>
    <td align="center">task</td>
    <td align="center">dataset</td>
    <td align="center">metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py">RotatedRetinaNet</a></td>
    <td align="center">Rotated Detection</td>
    <td align="center">DOTA-v1.0</td>
    <td align="center">mAP</td>
    <td align="center">0.698</td>
    <td align="center">0.698</td>
    <td align="center">0.698</td>
    <td align="center">0.697</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py">Oriented RCNN</a></td>
    <td align="center">Rotated Detection</td>
    <td align="center">DOTA-v1.0</td>
    <td align="center">mAP</td>
    <td align="center">0.756</td>
    <td align="center">0.756</td>
    <td align="center">0.758</td>
    <td align="center">0.730</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
</div>

## Notes

- As some datasets contain images with various resolutions in codebase like MMDet. The speed benchmark is gained through static configs in MMDeploy, while the performance benchmark is gained through dynamic ones.

- Some int8 performance benchmarks of TensorRT require Nvidia cards with tensor core, or the performance would drop heavily.

- DBNet uses the interpolate mode `nearest` in the neck of the model, which TensorRT-7 applies a quite different strategy from Pytorch. To make the repository compatible with TensorRT-7, we rewrite the neck to use the interpolate mode `bilinear` which improves final detection performance. To get the matched performance with Pytorch, TensorRT-8+ is recommended, which the interpolate methods are all the same as Pytorch.

- Mask AP of Mask R-CNN drops by 1% for the backend. The main reason is that the predicted masks are directly interpolated to original image in PyTorch, while they are at first interpolated to the preprocessed input image of the model and then to original image in other backends.

- MMPose models are tested with `flip_test` explicitly set to `False` in model configs.

- Some models might get low accuracy in fp16 mode. Please adjust the model to avoid value overflow.

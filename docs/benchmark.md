## Benchmark

### Backends
CPU: ncnn, ONNXRuntime
GPU: TensorRT, ppl.nn

### Platform
- Ubuntu 18.04
- Cuda 11.3
- TensorRT 7.2.3.4
- Docker 20.10.8
- NVIDIA tesla T4 tensor core GPU for TensorRT.

### Other settings
- Static graph
- Batch size 1
- Synchronize devices after each inference.
- We count the average inference performance of 100 images of the dataset.
- Warm up. For classification, we warm up 1010 iters. For other codebases, we warm up 10 iters.
- Input resolution varies for different datasets of different codebases. All inputs are real images except for mmediting because the dataset is not large enough.

### Latency benchmark
Users can directly test the performance through [how_to_measure_performance_of_models.md](docs/tutorials/how_to_measure_performance_of_models.md). And here is the benchmark in our environment.
<details>
<summary style="margin-left: 25px;">MMCls with 1x3x224x224 input</summary>
<div style="margin-left: 25px;">

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">2.97</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">336.90</span></td>
    <td class="tg-c3ow">1.26</td>
    <td class="tg-c3ow">791.89</td>
    <td class="tg-c3ow">1.21</td>
    <td class="tg-c3ow">829.66</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/resnet/resnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNeXt</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">4.31</td>
    <td class="tg-c3ow">231.93</td>
    <td class="tg-c3ow">1.42</td>
    <td class="tg-c3ow">703.42</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">727.42</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SE-ResNet</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">3.41</td>
    <td class="tg-c3ow">293.64</td>
    <td class="tg-c3ow">1.66</td>
    <td class="tg-c3ow">600.73</td>
    <td class="tg-c3ow">1.51</td>
    <td class="tg-c3ow">662.90</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/seresnet/seresnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ShuffleNetV2</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">727.94</td>
    <td class="tg-c3ow">1.19</td>
    <td class="tg-c3ow">841.36</td>
    <td class="tg-c3ow">1.13</td>
    <td class="tg-c3ow">883.47</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMediting with 1x3x32x32 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-lboi"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-nrix" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-lboi" rowspan="2"><span style="font-weight:400;font-style:normal">model config file</span></td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ESRGAN</td>
    <td class="tg-baqh">1x3x32x32</td>
    <td class="tg-c3ow">12.64</td>
    <td class="tg-c3ow">79.14</td>
    <td class="tg-c3ow">12.42</td>
    <td class="tg-c3ow">80.50</td>
    <td class="tg-c3ow">12.45</td>
    <td class="tg-c3ow">80.35</td>
    <td class="tg-0pky">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SRCNN</td>
    <td class="tg-baqh">1x3x32x32</td>
    <td class="tg-c3ow">0.70</td>
    <td class="tg-c3ow">1436.47</td>
    <td class="tg-c3ow">0.35</td>
    <td class="tg-c3ow">2836.62</td>
    <td class="tg-c3ow">0.26</td>
    <td class="tg-c3ow">3850.45</td>
    <td class="tg-0pky">$MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMSeg with 1x3x512x1024 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-c3ow">FCN</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">128.42</td>
    <td class="tg-c3ow">7.79</td>
    <td class="tg-c3ow">23.97</td>
    <td class="tg-c3ow">41.72</td>
    <td class="tg-c3ow">18.13</td>
    <td class="tg-c3ow">55.15</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">PSPNet</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">119.77</td>
    <td class="tg-c3ow">8.35</td>
    <td class="tg-c3ow">24.10</td>
    <td class="tg-c3ow">41.49</td>
    <td class="tg-c3ow">16.33</td>
    <td class="tg-c3ow">61.23</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DeepLabV3</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">226.75</td>
    <td class="tg-c3ow">4.41</td>
    <td class="tg-c3ow">31.80</td>
    <td class="tg-c3ow">31.45</td>
    <td class="tg-c3ow">19.85</td>
    <td class="tg-c3ow">50.38</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DeepLabV3+</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">151.25</td>
    <td class="tg-c3ow">6.61</td>
    <td class="tg-c3ow">47.03</td>
    <td class="tg-c3ow">21.26</td>
    <td class="tg-c3ow">50.38</td>
    <td class="tg-c3ow">26.67</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMDet with 1x3x800x1344 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-baqh">YOLOv3</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">94.08</td>
    <td class="tg-baqh">10.63</td>
    <td class="tg-baqh">24.90</td>
    <td class="tg-baqh">40.17</td>
    <td class="tg-baqh">24.87</td>
    <td class="tg-baqh">40.21</td>
    <td class="tg-0lax">$MMDET_DIR/configs/yolo/yolov3_d53_320_273e_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">SSD-Lite</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">14.91</td>
    <td class="tg-baqh">67.06</td>
    <td class="tg-baqh">8.92</td>
    <td class="tg-baqh">112.13</td>
    <td class="tg-baqh">8.65</td>
    <td class="tg-baqh">115.63</td>
    <td class="tg-0lax">$MMDET_DIR/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RetinaNet</td>
    <td class="tg-c3ow">1x3x800x1344</td>
    <td class="tg-c3ow">97.09</td>
    <td class="tg-c3ow">10.30</td>
    <td class="tg-c3ow">25.79</td>
    <td class="tg-c3ow">38.78</td>
    <td class="tg-c3ow">16.88</td>
    <td class="tg-c3ow">59.23</td>
    <td class="tg-0pky">$MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">FCOS</td>
    <td class="tg-c3ow">1x3x800x1344</td>
    <td class="tg-c3ow">84.06</td>
    <td class="tg-c3ow">11.90</td>
    <td class="tg-c3ow">23.15</td>
    <td class="tg-c3ow">43.20</td>
    <td class="tg-c3ow">17.68</td>
    <td class="tg-c3ow">56.57</td>
    <td class="tg-0pky">$MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">FSAF</td>
    <td class="tg-c3ow">1x3x800x1344</td>
    <td class="tg-c3ow">82.96</td>
    <td class="tg-c3ow">12.05</td>
    <td class="tg-c3ow">21.02</td>
    <td class="tg-c3ow">47.58</td>
    <td class="tg-c3ow">13.50</td>
    <td class="tg-c3ow">74.08</td>
    <td class="tg-0pky">$MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Faster-RCNN</td>
    <td class="tg-c3ow">1x3x800x1344</td>
    <td class="tg-c3ow">88.08</td>
    <td class="tg-c3ow">11.35</td>
    <td class="tg-c3ow">26.52</td>
    <td class="tg-c3ow">37.70</td>
    <td class="tg-c3ow">19.14</td>
    <td class="tg-c3ow">52.23</td>
    <td class="tg-0pky">$MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mask-RCNN</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">320.86 </td>
    <td class="tg-baqh">3.12</td>
    <td class="tg-baqh">241.32</td>
    <td class="tg-baqh">4.14</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-0lax">$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMOCR</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-baqh">DBNet</td>
    <td class="tg-baqh">1x3x640x640</td>
    <td class="tg-baqh">10.70</td>
    <td class="tg-baqh">93.43</td>
    <td class="tg-baqh">5.62</td>
    <td class="tg-baqh">177.78</td>
    <td class="tg-baqh">5.00</td>
    <td class="tg-baqh">199.85</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">CRNN</td>
    <td class="tg-baqh">1x1x32x32</td>
    <td class="tg-baqh">1.93 </td>
    <td class="tg-baqh">518.28</td>
    <td class="tg-baqh">1.40</td>
    <td class="tg-baqh">713.88</td>
    <td class="tg-baqh">1.36</td>
    <td class="tg-baqh">736.79</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textrecog/crnn/crnn_academic_dataset.py</td>
  </tr>
</tbody>
</table>
</div>
</details>
